import logging
import json
import os
import tempfile
import traceback
from functools import wraps, reduce
from inspect import isawaitable
from typing import Any, Callable, List, Optional, Text, Union

from sanic import Sanic, response
from sanic.request import Request
from sanic_cors import CORS
from sanic_jwt import Initialize, exceptions

import rasa
import rasa.utils.common
import rasa.utils.endpoints
import rasa.utils.io
from rasa.core.domain import InvalidDomain
from rasa.utils.endpoints import EndpointConfig
from rasa.constants import (
    MINIMUM_COMPATIBLE_VERSION,
    DEFAULT_MODELS_PATH,
    DEFAULT_DOMAIN_PATH,
    DOCS_BASE_URL,
)
from rasa.core import broker
from rasa.core.agent import load_agent, Agent
from rasa.core.channels.channel import (
    UserMessage,
    CollectingOutputChannel,
    OutputChannel,
)
from rasa.core.events import Event
from rasa.core.test import test
from rasa.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.core.utils import dump_obj_as_str_to_file, AvailableEndpoints
from rasa.model import get_model_subdirectories, fingerprint_from_path
from rasa.nlu.emulators.no_emulator import NoEmulator
from rasa.nlu.test import run_evaluation
from rasa.core.tracker_store import TrackerStore

logger = logging.getLogger(__name__)

OUTPUT_CHANNEL_QUERY_KEY = "output_channel"
USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL = "latest"


class ErrorResponse(Exception):
    def __init__(self, status, reason, message, details=None, help_url=None):
        self.error_info = {
            "version": rasa.__version__,
            "status": "failure",
            "message": message,
            "reason": reason,
            "details": details or {},
            "help": help_url,
            "code": status,
        }
        self.status = status


def _docs(sub_url: Text) -> Text:
    """Create a url to a subpart of the docs."""
    return DOCS_BASE_URL + sub_url


def ensure_loaded_agent(app: Sanic):
    """Wraps a request handler ensuring there is a loaded and usable agent."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not app.agent or not app.agent.is_ready():
                raise ErrorResponse(
                    409,
                    "Conflict",
                    "No agent loaded. To continue processing, a "
                    "model of a trained agent needs to be loaded.",
                    help_url=_docs("/user-guide/running-the-server/"),
                )

            return f(*args, **kwargs)

        return decorated

    return decorator


def requires_auth(app: Sanic, token: Optional[Text] = None) -> Callable[[Any], Any]:
    """Wraps a request handler with token authentication."""

    def decorator(f: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
        def conversation_id_from_args(args: Any, kwargs: Any) -> Optional[Text]:
            argnames = rasa.utils.common.arguments_of(f)

            try:
                sender_id_arg_idx = argnames.index("conversation_id")
                if "conversation_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["conversation_id"]
                if sender_id_arg_idx < len(args):
                    return args[sender_id_arg_idx]
                return None
            except ValueError:
                return None

        def sufficient_scope(request, *args: Any, **kwargs: Any) -> Optional[bool]:
            jwt_data = request.app.auth.extract_payload(request)
            user = jwt_data.get("user", {})

            username = user.get("username", None)
            role = user.get("role", None)

            if role == "admin":
                return True
            elif role == "user":
                conversation_id = conversation_id_from_args(args, kwargs)
                return conversation_id is not None and username == conversation_id
            else:
                return False

        @wraps(f)
        async def decorated(request: Request, *args: Any, **kwargs: Any) -> Any:

            provided = request.args.get("token", None)

            # noinspection PyProtectedMember
            if token is not None and provided == token:
                result = f(request, *args, **kwargs)
                if isawaitable(result):
                    result = await result
                return result
            elif app.config.get("USE_JWT") and request.app.auth.is_authenticated(
                    request
            ):
                if sufficient_scope(request, *args, **kwargs):
                    result = f(request, *args, **kwargs)
                    if isawaitable(result):
                        result = await result
                    return result
                raise ErrorResponse(
                    403,
                    "NotAuthorized",
                    "User has insufficient permissions.",
                    help_url=_docs(
                        "/user-guide/running-the-server/#security-considerations"
                    ),
                )
            elif token is None and app.config.get("USE_JWT") is None:
                # authentication is disabled
                result = f(request, *args, **kwargs)
                if isawaitable(result):
                    result = await result
                return result
            raise ErrorResponse(
                401,
                "NotAuthenticated",
                "User is not authenticated.",
                help_url=_docs(
                    "/user-guide/running-the-server/#security-considerations"
                ),
            )

        return decorated

    return decorator


def event_verbosity_parameter(
        request: Request, default_verbosity: EventVerbosity
) -> EventVerbosity:
    event_verbosity_str = request.args.get(
        "include_events", default_verbosity.name
    ).upper()
    try:
        return EventVerbosity[event_verbosity_str]
    except KeyError:
        enum_values = ", ".join([e.name for e in EventVerbosity])
        raise ErrorResponse(
            400,
            "BadRequest",
            "Invalid parameter value for 'include_events'. "
            "Should be one of {}".format(enum_values),
            {"parameter": "include_events", "in": "query"},
        )


def obtain_tracker_store(agent: "Agent", conversation_id: Text) -> DialogueStateTracker:
    tracker = agent.tracker_store.get_or_create_tracker(conversation_id)
    if not tracker:
        raise ErrorResponse(
            409,
            "Conflict",
            "Could not retrieve tracker with id '{}'. Most likely "
            "because there is no domain set on the agent.".format(conversation_id),
        )
    return tracker


def validate_request_body(request: Request, error_message: Text):
    if not request.body:
        raise ErrorResponse(400, "BadRequest", error_message)


async def authenticate(request: Request):
    raise exceptions.AuthenticationFailed(
        "Direct JWT authentication not supported. You should already have "
        "a valid JWT from an authentication provider, Rasa will just make "
        "sure that the token is valid, but not issue new tokens."
    )


def _create_emulator(mode: Optional[Text]) -> NoEmulator:
    """Create emulator for specified mode.
    If no emulator is specified, we will use the Rasa NLU format."""

    if mode is None:
        return NoEmulator()
    elif mode.lower() == "wit":
        from rasa.nlu.emulators.wit import WitEmulator

        return WitEmulator()
    elif mode.lower() == "luis":
        from rasa.nlu.emulators.luis import LUISEmulator

        return LUISEmulator()
    elif mode.lower() == "dialogflow":
        from rasa.nlu.emulators.dialogflow import DialogflowEmulator

        return DialogflowEmulator()
    else:
        raise ErrorResponse(
            400,
            "BadRequest",
            "Invalid parameter value for 'emulation_mode'. "
            "Should be one of 'WIT', 'LUIS', 'DIALOGFLOW'.",
            {"parameter": "emulation_mode", "in": "query"},
        )


async def _load_agent(
        model_path: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        endpoints: Optional[AvailableEndpoints] = None,
) -> Agent:
    try:
        tracker_store = None
        generator = None
        action_endpoint = None

        if endpoints:
            _broker = broker.from_endpoint_config(endpoints.event_broker)
            tracker_store = TrackerStore.find_tracker_store(
                None, endpoints.tracker_store, _broker
            )
            generator = endpoints.nlg
            action_endpoint = endpoints.action

        loaded_agent = await load_agent(
            model_path,
            model_server,
            remote_storage,
            generator=generator,
            tracker_store=tracker_store,
            action_endpoint=action_endpoint,
        )
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise ErrorResponse(
            500, "LoadingError", "An unexpected error occurred. Error: {}".format(e)
        )

    if not loaded_agent:
        raise ErrorResponse(
            400,
            "BadRequest",
            "Agent with name '{}' could not be loaded.".format(model_path),
            {"parameter": "model", "in": "query"},
        )

    return loaded_agent


def add_root_route(app: Sanic):
    @app.get("/")
    async def hello(request: Request):
        """Check if the server is running and responds with the version."""
        return response.text("Hello from Rasa: " + rasa.__version__)


def create_app(
        agent: Optional["Agent"] = None,
        cors_origins: Union[Text, List[Text]] = "*",
        auth_token: Optional[Text] = None,
        jwt_secret: Optional[Text] = None,
        jwt_method: Text = "HS256",
        endpoints: Optional[AvailableEndpoints] = None,
):
    """Class representing a Rasa HTTP server."""

    app = Sanic(__name__)
    app.config.RESPONSE_TIMEOUT = 60 * 60

    CORS(
        app, resources={r"/*": {"origins": cors_origins or ""}}, automatic_options=True
    )

    # Setup the Sanic-JWT extension
    if jwt_secret and jwt_method:
        # since we only want to check signatures, we don't actually care
        # about the JWT method and set the passed secret as either symmetric
        # or asymmetric key. jwt lib will choose the right one based on method
        app.config["USE_JWT"] = True
        Initialize(
            app,
            secret=jwt_secret,
            authenticate=authenticate,
            algorithm=jwt_method,
            user_id="username",
        )

    app.agent = agent

    @app.exception(ErrorResponse)
    async def handle_error_response(request: Request, exception: ErrorResponse):
        return response.json(exception.error_info, status=exception.status)

    add_root_route(app)

    @app.get("/version")
    async def version(request: Request):
        """Respond with the version number of the installed Rasa."""

        return response.json(
            {
                "version": rasa.__version__,
                "minimum_compatible_version": MINIMUM_COMPATIBLE_VERSION,
            }
        )

    @app.get("/status")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def status(request: Request):
        """Respond with the model name and the fingerprint of that model."""

        return response.json(
            {
                "model_file": app.agent.model_directory,
                "fingerprint": fingerprint_from_path(app.agent.model_directory),
            }
        )

    @app.get("/conversations/<conversation_id>/tracker")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def retrieve_tracker(request: Request, conversation_id: Text):
        """Get a dump of a conversation's tracker including its events."""
        if not app.agent.tracker_store:
            raise ErrorResponse(
                409,
                "Conflict",
                "No tracker store available. Make sure to "
                "configure a tracker store when starting "
                "the server.",
            )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        until_time = rasa.utils.endpoints.float_arg(request, "until")

        tracker = obtain_tracker_store(app.agent, conversation_id)

        try:
            if until_time is not None:
                tracker = tracker.travel_back_in_time(until_time)

            state = tracker.current_state(verbosity)
            return response.json(state)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/conversations/<conversation_id>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def append_events(request: Request, conversation_id: Text):
        """Append a list of events to the state of a conversation"""
        validate_request_body(
            request,
            "You must provide events in the request body in order to append them"
            "to the state of a conversation.",
        )

        events = request.json
        if not isinstance(events, list):
            events = [events]

        events = [Event.from_parameters(event) for event in events]
        events = [event for event in events if event]

        if not events:
            logger.warning(
                "Append event called, but could not extract a valid event. "
                "Request JSON: {}".format(request.json)
            )
            raise ErrorResponse(
                400,
                "BadRequest",
                "Couldn't extract a proper event from the request body.",
                {"parameter": "", "in": "body"},
            )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        tracker = obtain_tracker_store(app.agent, conversation_id)

        try:
            for event in events:
                tracker.update(event, app.agent.domain)

            app.agent.tracker_store.save(tracker)

            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.put("/conversations/<conversation_id>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def replace_events(request: Request, conversation_id: Text):
        """Use a list of events to set a conversations tracker to a state."""
        validate_request_body(
            request,
            "You must provide events in the request body to set the sate of the "
            "conversation tracker.",
        )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            tracker = DialogueStateTracker.from_dict(
                conversation_id, request.json, app.agent.domain.slots
            )

            # will override an existing tracker with the same id!
            app.agent.tracker_store.save(tracker)
            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.get("/conversations/<conversation_id>/story")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def retrieve_story(request: Request, conversation_id: Text):
        """Get an end-to-end story corresponding to this conversation."""
        if not app.agent.tracker_store:
            raise ErrorResponse(
                409,
                "Conflict",
                "No tracker store available. Make sure to "
                "configure a tracker store when starting "
                "the server.",
            )

        # retrieve tracker and set to requested state
        tracker = obtain_tracker_store(app.agent, conversation_id)

        until_time = rasa.utils.endpoints.float_arg(request, "until")

        try:
            if until_time is not None:
                tracker = tracker.travel_back_in_time(until_time)

            # dump and return tracker
            state = tracker.export_stories(e2e=True)
            return response.text(state)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/conversations/<conversation_id>/execute")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def execute_action(request: Request, conversation_id: Text):
        request_params = request.json

        action_to_execute = request_params.get("name", None)

        if not action_to_execute:
            raise ErrorResponse(
                400,
                "BadRequest",
                "Name of the action not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        policy = request_params.get("policy", None)
        confidence = request_params.get("confidence", None)
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            tracker = obtain_tracker_store(app.agent, conversation_id)
            output_channel = _get_output_channel(request, tracker)
            await app.agent.execute_action(
                conversation_id, action_to_execute, output_channel, policy, confidence
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

        tracker = obtain_tracker_store(app.agent, conversation_id)
        state = tracker.current_state(verbosity)

        response_body = {"tracker": state}

        if isinstance(output_channel, CollectingOutputChannel):
            response_body["messages"] = output_channel.messages

        return response.json(response_body)

    @app.post("/conversations/<conversation_id>/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def predict(request: Request, conversation_id: Text):
        try:
            # Fetches the appropriate bot response in a json format
            responses = app.agent.predict_next(conversation_id)
            responses["scores"] = sorted(
                responses["scores"], key=lambda k: (-k["score"], k["action"])
            )
            return response.json(responses)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/conversations/<conversation_id>/messages")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def add_message(request: Request, conversation_id: Text):
        validate_request_body(
            request,
            "No message defined in request body. Add a message to the request body in "
            "order to add it to the tracker.",
        )

        request_params = request.json

        message = request_params.get("text")
        sender = request_params.get("sender")
        parse_data = request_params.get("parse_data")

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        # TODO: implement for agent / bot
        if sender != "user":
            raise ErrorResponse(
                400,
                "BadRequest",
                "Currently, only user messages can be passed to this endpoint. "
                "Messages of sender '{}' cannot be handled.".format(sender),
                {"parameter": "sender", "in": "body"},
            )

        try:
            user_message = UserMessage(message, None, conversation_id, parse_data)
            tracker = await app.agent.log_message(user_message)
            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/model/train")
    @requires_auth(app, auth_token)
    async def train(request: Request):
        """Train a Rasa Model."""
        from rasa.train import train_async

        validate_request_body(
            request,
            "You must provide training data in the request body in order to "
            "train your model.",
        )

        rjs = request.json
        validate_request(rjs)

        # create a temporary directory to store config, domain and
        # training data
        temp_dir = tempfile.mkdtemp()

        config_path = os.path.join(temp_dir, "config.yml")
        dump_obj_as_str_to_file(config_path, rjs["config"])

        if "nlu" in rjs:
            nlu_path = os.path.join(temp_dir, "nlu.md")
            dump_obj_as_str_to_file(nlu_path, rjs["nlu"])

        if "stories" in rjs:
            stories_path = os.path.join(temp_dir, "stories.md")
            dump_obj_as_str_to_file(stories_path, rjs["stories"])

        domain_path = DEFAULT_DOMAIN_PATH
        if "domain" in rjs:
            domain_path = os.path.join(temp_dir, "domain.yml")
            dump_obj_as_str_to_file(domain_path, rjs["domain"])

        try:
            model_path = await train_async(
                domain=domain_path,
                config=config_path,
                training_files=temp_dir,
                output_path=rjs.get("out", DEFAULT_MODELS_PATH),
                force_training=rjs.get("force", False),
            )

            filename = os.path.basename(model_path) if model_path else None

            return await response.file(
                model_path, filename=filename, headers={"filename": filename}
            )
        except InvalidDomain as e:
            raise ErrorResponse(
                400,
                "InvalidDomainError",
                "Provided domain file is invalid. Error: {}".format(e),
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "TrainingError",
                "An unexpected error occurred during training. Error: {}".format(e),
            )

    def validate_request(rjs):
        if "config" not in rjs:
            raise ErrorResponse(
                400,
                "BadRequest",
                "The training request is missing the required key `config`.",
                {"parameter": "config", "in": "body"},
            )

        if "nlu" not in rjs and "stories" not in rjs:
            raise ErrorResponse(
                400,
                "BadRequest",
                "To train a Rasa model you need to specify at least one type of "
                "training data. Add `nlu` and/or `stories` to the request.",
                {"parameters": ["nlu", "stories"], "in": "body"},
            )

        if "stories" in rjs and "domain" not in rjs:
            raise ErrorResponse(
                400,
                "BadRequest",
                "To train a Rasa model with story training data, you also need to "
                "specify the `domain`.",
                {"parameter": "domain", "in": "body"},
            )

    @app.post("/model/test/stories")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def evaluate_stories(request: Request):
        """Evaluate stories against the currently loaded model."""
        validate_request_body(
            request,
            "You must provide some stories in the request body in order to "
            "evaluate your model.",
        )

        stories = rasa.utils.io.create_temporary_file(request.body, mode="w+b")
        use_e2e = rasa.utils.endpoints.bool_arg(request, "e2e", default=False)

        try:
            evaluation = await test(stories, app.agent, e2e=use_e2e)
            return response.json(evaluation)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "TestingError",
                "An unexpected error occurred during evaluation. Error: {}".format(e),
            )

    @app.post("/model/test/intents")
    @requires_auth(app, auth_token)
    async def evaluate_intents(request: Request):
        """Evaluate intents against a Rasa model."""
        validate_request_body(
            request,
            "You must provide some nlu data in the request body in order to "
            "evaluate your model.",
        )

        eval_agent = app.agent

        model_path = request.args.get("model", None)
        if model_path:
            model_server = app.agent.model_server
            if model_server is not None:
                model_server.url = model_path
            eval_agent = await _load_agent(
                model_path, model_server, app.agent.remote_storage
            )

        nlu_data = rasa.utils.io.create_temporary_file(request.body, mode="w+b")
        data_path = os.path.abspath(nlu_data)

        if not os.path.exists(eval_agent.model_directory):
            raise ErrorResponse(409, "Conflict", "Loaded model file not found.")

        model_directory = eval_agent.model_directory
        _, nlu_model = get_model_subdirectories(model_directory)

        try:
            evaluation = run_evaluation(data_path, nlu_model)
            return response.json(evaluation)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "TestingError",
                "An unexpected error occurred during evaluation. Error: {}".format(e),
            )

    @app.post("/model/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def tracker_predict(request: Request):
        """ Given a list of events, predicts the next action"""
        validate_request_body(
            request,
            "No events defined in request_body. Add events to request body in order to "
            "predict the next action.",
        )

        sender_id = UserMessage.DEFAULT_SENDER_ID
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        request_params = request.json

        try:
            tracker = DialogueStateTracker.from_dict(
                sender_id, request_params, app.agent.domain.slots
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                400,
                "BadRequest",
                "Supplied events are not valid. {}".format(e),
                {"parameter": "", "in": "body"},
            )

        try:
            policy_ensemble = app.agent.policy_ensemble
            probabilities, policy = policy_ensemble.probabilities_using_best_policy(
                tracker, app.agent.domain
            )

            scores = [
                {"action": a, "score": p}
                for a, p in zip(app.agent.domain.action_names, probabilities)
            ]

            return response.json(
                {
                    "scores": scores,
                    "policy": policy,
                    "tracker": tracker.current_state(verbosity),
                }
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "PredictionError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/model/parse")
    @requires_auth(app, auth_token)
    async def parse(request: Request):
        validate_request_body(
            request,
            "No text message defined in request_body. Add text message to request body "
            "in order to obtain the intent and extracted entities.",
        )
        emulation_mode = request.args.get("emulation_mode")
        emulator = _create_emulator(emulation_mode)

        try:
            data = emulator.normalise_request_json(request.json)
            try:
                data['text'] = data['text'].lower().replace("\'", "")
                parsed_data = await app.agent.parse_message_using_nlu_interpreter(
                    data.get("text")
                )
            except Exception as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    400,
                    "ParsingError",
                    "An unexpected error occurred. Error: {}".format(e),
                )
            response_data = emulator.normalise_response_json(parsed_data)
            print(response_data)
            if response_data['intent']['confidence'] >= 0.70:
                narrowedEntity = entitySerializer(response_data['entities'])
                entMap = entityMapper(narrowedEntity, response_data['intent']['name'], response_data['text'])
                response_data['slotvalues'] = entMap
                response_data['didYouMean'] = False
                response_data['intent'] = response_data['intent']['name'] if response_data['intent'][
                                                                                 'name'] != "SeriesIntent" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "search_title" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "search_subject" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "search_author" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "searchSubject" else "SearchIntent"
                response_data['reqtype'] = respFinder(response_data['intent'])
            else:
                response_data['intent'] = response_data['intent']['name'] if response_data['intent'][
                                                                                 'name'] != "SeriesIntent" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "search_title" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "search_subject" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "search_author" and \
                                                                             response_data['intent'][
                                                                                 'name'] != "searchSubject" else "SearchIntent"
                respData = []
                for each in response_data["intent_ranking"]:
                    each["name"] = each['name'] if each['name'] != "SeriesIntent" and \
                                                   each['name'] != "search_title" and \
                                                   each['name'] != "search_subject" and \
                                                   each['name'] != "search_author" and \
                                                   each['name'] != "searchSubject" else "SearchIntent"
                    respData.append(each)
                response_data['slotvalues'] = didYouMean(response_data['entities'])
                response_data['didYouMean'] = True
                response_data['reqtype'] = respFinder(response_data['intent'])
                response_data['intent_ranking'] = respData
                del response_data['entities']
            return response.json(response_data)

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500, "ParsingError", "An unexpected error occurred. Error: {}".format(e)
            )

    def respFinder(intent):
        if intent == "AMAZON.StopIntent":
            return 'SessionEndedRequest'
        else:
            return 'IntentRequest'

    def didYouMean(entity):
        entityArray = []
        for ele in entity:
            data = {}
            if ele['entity'] == 'time':
                if 'from' in ele['value']:
                    ele['value'] = json.dumps(ele['value']).replace("\'", "\"", -1)
                    datamap = json.loads(ele['value'])
                    tempMap = {}
                    tempMap['name'] = 'from'
                    fromDate = datamap['from'].split("T")
                    tempMap['value'] = fromDate[0]
                    entityArray.append(tempMap)
                    tempMap = {}
                    tempMap['name'] = 'to'
                    todate = datamap['to'].split("T")
                    tempMap['value'] = todate[0]
                    entityArray.append(tempMap)
                else:
                    date = ele["value"].split("T")
                    data["name"] = 'date'
                    data["value"] = date[0]
                    entityArray.append(data)
            else:
                data['name'] = ele['entity']
                data['value'] = ele['value']
                entityArray.append(data)
        return entityArray

    def entityMapper(entMap, intent, utterence):
        print(entMap, intent, utterence)
        intent = intent.lower()
        conditionMap = {}
        entityMap = {}
        entityArray = []
        count = 0
        print(entMap, intent, utterence)
        newMap = {}
        for data in entMap:
            newMap[data['name']] = data['value']
        if newMap.__contains__("WORK_OF_ART") and newMap.__contains__("subject"):
            print("into special condition")
            newMap.__delitem__("WORK_OF_ART")
            entMap = []
            resMap = {}
            for data, value in newMap.items():
                resMap['name'] = data
                resMap['value'] = value
                entMap.append(resMap)
                resMap = {}
        if intent == "searchintent" or intent == "cancelholdintent" or intent == "renewintent" or intent == "listcheckOutintent" or intent == "listholdintent":
            contentMap = {}
            for data in entMap:
                contentMap[data['name']] = data['value']
            print("entity map value is ", entMap)
            for data in entMap:
                if data["name"] == "WORK_OF_ART":
                    data["name"] = "stitle"
                    data["value"] = data["value"].lower().replace("search for a book", "").replace(
                        "search for the book", "").replace("serach for title", "").replace("search for a title",
                                                                                           "").replace(
                        "serach for the title", "")
                    if data["value"] != "":
                        if 'filterphrase' in contentMap and contentMap['WORK_OF_ART'] == contentMap['filterphrase']:
                            pass
                        else:
                            entityArray.append(data)
                            conditionMap["stitle"] = data["value"]
                    print("Entity array after work of art is ", entityArray)
                elif data["name"] == "person":
                    print(data)
                    if "stitle" in conditionMap:
                        if data["value"] != conditionMap["stitle"]:
                            data["name"] = "sauthor"
                            entityArray.append(data)
                    else:
                        data["name"] = "sauthor"
                        entityArray.append(data)
                elif data["name"] == "sBook":
                    if "stitle" not in conditionMap:
                        data["name"] = "stitle"
                        entityArray.append(data)
                        conditionMap["stitle"] = data["value"]
                elif data["name"] == "series":
                    if "stitle" not in conditionMap:
                        data["name"] = "stitle"
                        entityArray.append(data)
                        conditionMap["stitle"] = data["value"]
                elif data["name"] == "sbook":
                    if "stitle" not in conditionMap:
                        if len(data["value"]) <= 50:
                            data["name"] = "stitle"
                            entityArray.append(data)
                            conditionMap["stitle"] = data["value"]
                        else:
                            if "subject" not in conditionMap:
                                data["name"] = "subject"
                                data["name"] = data["name"].lower()
                                entityArray.append(data)
                                conditionMap["subject"] = data["value"]
                    else:
                        print(entityArray)
                        entityArray.pop(0)
                        conditionMap["stitle"] = data["value"].lower()
                        data["value"] = data["value"].lower()
                        data["name"] = "stitle"
                        entityArray.append(data)
                        print(entityArray)
                elif data["name"] == "subject":
                    print("Entity array before subject is ", entityArray)
                    if "subject" not in conditionMap:
                        data["name"] = data["name"].lower()
                        entityArray.append(data)
                        conditionMap["subject"] = data["value"]
                elif data["name"] == "filterphrase":
                    if data["value"] in conditionMap.values():
                        pass
                    elif conditionMap.__contains__("stitle"):
                        if data["value"] == conditionMap["stitle"]:
                            pass
                        else:
                            data["name"] = data["name"].lower()
                            entityArray.append(data)
                    else:
                        data["name"] = data["name"].lower()
                        entityArray.append(data)
                elif data["name"] == "sauthor":
                    entityArray.append(data)
                elif data["name"] == "mtype":
                    entityArray.append(data)
                elif data["name"] == "language":
                    data["name"] = 'lang'
                    entityArray.append(data)
                elif data["name"] == 'type' or data["name"] == 'timeline' or data[
                    "name"] == 'mtype' or data["name"] == 'renew' or data["name"] == 'renewAll' or data[
                    "name"] == "cancelhold" or data["name"] == 'type' or data['name'] == 'reserve' or data[
                    'name'] == 'lang' or data['name'] == 'library':
                    data["name"] = data["name"].lower()
                    entityArray.append(data)
                elif data["name"] == "pubyear":
                    if conditionMap.__contains__("pubyear"):
                        pass
                    else:
                        conditionMap['pubyear'] = data['value']
                        entityArray.append(data)
                elif data["name"] == "number":
                    if conditionMap.__contains__("pubyear"):
                        pass
                    else:
                        conditionMap['pubyear'] = data['value']
                        data["name"] = "pubyear"
                        entityArray.append(data)
                else:
                    pass
        elif intent == "seriesintent":
            for data in entMap:
                if data["name"] == "WORK_OF_ART":
                    data["name"] = "sseries"
                    data["value"] = data["value"].lower().replace("search for a book", "").replace(
                        "search for the book", "").replace("serach for title", "").replace("search for a title",
                                                                                           "").replace(
                        "serach for the title", "")
                    if data["value"] != "":
                        entityArray.append(data)
                        conditionMap["sseries"] = data["value"]
                        print("Entity array after work of art is ", entityArray)
                elif data["name"] == "person":
                    if "sseries" in conditionMap:
                        if data["value"] != conditionMap["sseries"]:
                            data["name"] = "sauthor"
                            entityArray.append(data)
                    else:
                        data["name"] = "sauthor"
                        entityArray.append(data)
                elif data["name"] == "series":
                    if "sseries" not in conditionMap:
                        data["name"] = "sseries"
                        entityArray.append(data)
                        conditionMap["sseries"] = data["value"]
                    else:
                        entityArray.pop(0)
                        conditionMap["sseries"] = data["value"].lower()
                        data["value"] = data["value"].lower()
                        data["name"] = "sseries"
                        entityArray.append(data)
                elif data["name"] == "sbook" or data["name"] == "sBook":
                    if "sseries" not in conditionMap:
                        data["name"] = "sseries"
                        entityArray.append(data)
                        conditionMap["sseries"] = data["value"]
                elif data["name"] == "subject":
                    print("Entity array before subject is ", entityArray)
                    if "subject" not in conditionMap:
                        data["name"] = data["name"].lower()
                        entityArray.append(data)
                        conditionMap["subject"] = data["value"]
                elif data["name"] == "filterphrase":
                    if data["value"] in conditionMap.values():
                        pass
                    elif conditionMap.__contains__("sseries"):
                        if data["value"] == conditionMap["sseries"]:
                            pass
                        else:
                            data["name"] = data["name"].lower()
                            entityArray.append(data)
                    else:
                        data["name"] = data["name"].lower()
                        entityArray.append(data)
                elif data["name"] == "sauthor":
                    entityArray.append(data)
                elif data["name"] == 'type' or data["name"] == 'timeline' or data["name"] == 'mtype' or data[
                    "name"] == 'renew' or data["name"] == 'renewAll' or data["name"] == "cancelhold":
                    data["name"] = data["name"].lower()
                    entityArray.append(data)
                elif data["name"] == "pubyear":
                    if conditionMap.__contains__("pubyear"):
                        pass
                    else:
                        conditionMap['pubyear'] = data['value']
                        entityArray.append(data)
                elif data["name"] == "number":
                    if conditionMap.__contains__("pubyear"):
                        pass
                    else:
                        conditionMap['pubyear'] = data['value']
                        data["name"] = "pubyear"
                        entityArray.append(data)
                else:
                    pass
        elif intent == "search_title":
            customMap = {}
            customMap["name"] = "type"
            customMap["value"] = "title"
            entityArray.append(customMap)
            for data in entMap:
                for data in entMap:
                    if data["name"] == "WORK_OF_ART":
                        data["name"] = "stitle"
                        data["value"] = data["value"].lower().replace("search for a book", "").replace(
                            "search for the book", "").replace("serach for title", "").replace("search for a title",
                                                                                               "").replace(
                            "serach for the title", "")
                        if data["value"] != "":
                            if 'filterphrase' in conditionMap and conditionMap['WORK_OF_ART'] == conditionMap[
                                'filterphrase']:
                                pass
                            else:
                                entityArray.append(data)
                                conditionMap["stitle"] = data["value"]
                        print("Entity array after work of art is ", entityArray)
                    elif data["name"] == "person":
                        if "stitle" in conditionMap:
                            if data["value"] != conditionMap["stitle"]:
                                data["name"] = "sauthor"
                                entityArray.append(data)
                        else:
                            data["name"] = "sauthor"
                            entityArray.append(data)
                    elif data["name"] == "sBook":
                        if "stitle" not in conditionMap:
                            data["name"] = "stitle"
                            entityArray.append(data)
                            conditionMap["stitle"] = data["value"]
                    elif data["name"] == "sbook":
                        if "stitle" not in conditionMap:
                            if len(data["value"]) <= 50:
                                data["name"] = "stitle"
                                entityArray.append(data)
                                conditionMap["stitle"] = data["value"]
                            else:
                                if "subject" not in conditionMap:
                                    data["name"] = "subject"
                                    data["name"] = data["name"].lower()
                                    entityArray.append(data)
                                    conditionMap["subject"] = data["value"]
                        else:
                            print(entityArray)
                            entityArray.pop(0)
                            conditionMap["stitle"] = data["value"].lower()
                            data["value"] = data["value"].lower()
                            data["name"] = "stitle"
                            entityArray.append(data)
                            print(entityArray)
                    elif data["name"] == "subject":
                        print("Entity array before subject is ", entityArray)
                        if "subject" not in conditionMap:
                            data["name"] = data["name"].lower()
                            entityArray.append(data)
                            conditionMap["subject"] = data["value"]
        elif intent == "search_author":
            customMap = {}
            customMap["name"] = "type"
            customMap["value"] = "author"
            entityArray.append(customMap)
            for data in entMap:
                for data in entMap:
                    if data["name"] == "WORK_OF_ART":
                        data["name"] = "stitle"
                        data["value"] = data["value"].lower().replace("search for a book", "").replace(
                            "search for the book", "").replace("serach for title", "").replace("search for a title",
                                                                                               "").replace(
                            "serach for the title", "")
                        if data["value"] != "":
                            if 'filterphrase' in conditionMap and conditionMap['WORK_OF_ART'] == conditionMap[
                                'filterphrase']:
                                pass
                            else:
                                entityArray.append(data)
                                conditionMap["stitle"] = data["value"]
                        print("Entity array after work of art is ", entityArray)
                    elif data["name"] == "person":
                        if "stitle" in conditionMap:
                            if data["value"] != conditionMap["stitle"]:
                                data["name"] = "sauthor"
                                entityArray.append(data)
                        else:
                            data["name"] = "sauthor"
                            entityArray.append(data)
                    elif data["name"] == "sBook":
                        if "stitle" not in conditionMap:
                            data["name"] = "stitle"
                            entityArray.append(data)
                            conditionMap["stitle"] = data["value"]
                    elif data["name"] == "sbook":
                        if "stitle" not in conditionMap:
                            if len(data["value"]) <= 50:
                                data["name"] = "stitle"
                                entityArray.append(data)
                                conditionMap["stitle"] = data["value"]
                            else:
                                if "subject" not in conditionMap:
                                    data["name"] = "subject"
                                    data["name"] = data["name"].lower()
                                    entityArray.append(data)
                                    conditionMap["subject"] = data["value"]
                        else:
                            print(entityArray)
                            entityArray.pop(0)
                            conditionMap["stitle"] = data["value"].lower()
                            data["value"] = data["value"].lower()
                            data["name"] = "stitle"
                            entityArray.append(data)
                            print(entityArray)
                    elif data["name"] == "subject":
                        print("Entity array before subject is ", entityArray)
                        if "subject" not in conditionMap:
                            data["name"] = data["name"].lower()
                            entityArray.append(data)
                            conditionMap["subject"] = data["value"]
        elif intent == "search_subject":
            customMap = {}
            customMap["name"] = "type"
            customMap["value"] = "subject"
            entityArray.append(customMap)
            for data in entMap:
                for data in entMap:
                    if data["name"] == "WORK_OF_ART":
                        data["name"] = "stitle"
                        data["value"] = data["value"].lower().replace("search for a book", "").replace(
                            "search for the book", "").replace("serach for title", "").replace("search for a title",
                                                                                               "").replace(
                            "serach for the title", "")
                        if data["value"] != "":
                            if 'filterphrase' in conditionMap and conditionMap['WORK_OF_ART'] == conditionMap[
                                'filterphrase']:
                                pass
                            else:
                                entityArray.append(data)
                                conditionMap["stitle"] = data["value"]
                        print("Entity array after work of art is ", entityArray)
                    elif data["name"] == "person":
                        if "stitle" in conditionMap:
                            if data["value"] != conditionMap["stitle"]:
                                data["name"] = "sauthor"
                                entityArray.append(data)
                        else:
                            data["name"] = "sauthor"
                            entityArray.append(data)
                    elif data["name"] == "sBook":
                        if "stitle" not in conditionMap:
                            data["name"] = "stitle"
                            entityArray.append(data)
                            conditionMap["stitle"] = data["value"]
                    elif data["name"] == "sbook":
                        if "stitle" not in conditionMap:
                            if len(data["value"]) <= 50:
                                data["name"] = "stitle"
                                entityArray.append(data)
                                conditionMap["stitle"] = data["value"]
                            else:
                                if "subject" not in conditionMap:
                                    data["name"] = "subject"
                                    data["name"] = data["name"].lower()
                                    entityArray.append(data)
                                    conditionMap["subject"] = data["value"]
                        else:
                            print(entityArray)
                            entityArray.pop(0)
                            conditionMap["stitle"] = data["value"].lower()
                            data["value"] = data["value"].lower()
                            data["name"] = "stitle"
                            entityArray.append(data)
                            print(entityArray)
                    elif data["name"] == "subject":
                        print("Entity array before subject is ", entityArray)
                        if "subject" not in conditionMap:
                            data["name"] = data["name"].lower()
                            entityArray.append(data)
                            conditionMap["subject"] = data["value"]
        elif intent == "checkedoutintent":
            for data in entMap:
                if data["name"] == "checkedout" or data["name"] == "mtype":
                    entityArray.append(data)
                else:
                    pass
        elif intent == "eventintent":
            for data in entMap:
                if data["name"] == "person":
                    if "present" in utterence and "organize" in utterence:
                        orgsplit = utterence.split("organize")
                        presplit = utterence.split("present")
                        secondkey = ""
                        if count == 0:
                            if len(orgsplit[0]) < len(presplit[0]):
                                data["name"] = 'organizer'
                            else:
                                data["name"] = 'presenter'
                            data["value"] = data["value"].replace(" in", "", -1).replace(" at", "", -1)
                            entityArray.append(data)
                            count = count + 1
                        elif count == 1:
                            if len(orgsplit[0]) < len(presplit[0]):
                                data["name"] = 'presenter'
                            else:
                                data["name"] = 'organizer'
                            data["value"] = data["value"].replace(" in", "", -1).replace(" at", "", -1)
                            entityArray.append(data)
                            count = 0
                    elif "present" in utterence:
                        data["name"] = "presenter"
                        data["value"] = data["value"].replace(" in", "", -1).replace(" at", "", -1)
                        entityArray.append(data)
                    elif "organize" in utterence:
                        data["name"] = "organizer"
                        data["value"] = data["value"].replace(" in", "", -1).replace(" at", "", -1)
                        entityArray.append(data)
                elif data["name"] == "time":
                    if 'from' in data['value']:
                        print("*********************************************************")
                        data['value'] = data['value'].replace("\'", "\"", -1)
                        datamap = json.loads(data['value'])
                        tempMap = {}
                        tempMap['name'] = 'from'
                        fromDate = datamap['from'].split("T")
                        tempMap['value'] = fromDate[0]
                        entityArray.append(tempMap)
                        tempMap = {}
                        tempMap['name'] = 'to'
                        todate = datamap['to'].split("T")
                        tempMap['value'] = todate[0]
                        entityArray.append(tempMap)
                    else:
                        date = data["value"].split("T")
                        print(type(data["value"]))
                        print(date)
                        data["name"] = 'edate'
                        data["value"] = date[0]
                        entityArray.append(data)
                        tempMap = {}
                        if 'week' in utterence:
                            tempMap['name'] = 'dateFilter'
                            tempMap['value'] = 'week'
                            entityArray.append(tempMap)
                        elif 'month' in utterence:
                            tempMap['name'] = 'dateFilter'
                            tempMap['value'] = 'month'
                            entityArray.append(tempMap)
                        elif 'year' in utterence:
                            tempMap['name'] = 'dateFilter'
                            tempMap['value'] = 'year'
                            entityArray.append(tempMap)
                    conditionMap["edate"] = data["value"]
                elif data["name"].lower() == 'date':
                    data["name"] = 'date'
                elif data["name"] == "edate":
                    if "edate" not in conditionMap:
                        entityArray.append(data)
                elif data["name"] == "ORG" or data["name"] == "libname":
                    data["name"] = "library"
                    entityArray.append(data)
                elif data["name"] == "lang":
                    data["name"] = "language"
                    entityArray.append(data)
                elif data["name"] == "subject" or data["name"] == "title" or data["name"] == "program":
                    if 'searchQuery' not in conditionMap:
                        if 'on' in utterence:
                            data["name"] = 'title'
                            entityArray.append(data)
                            conditionMap['searchQuery'] = True
                        else:
                            data["name"] = 'program'
                            entityArray.append(data)
                            conditionMap['searchQuery'] = True
                    else:
                        pass
                elif data["name"] == "language" or data["name"] == "library" or data["name"] == "category":
                    entityArray.append(data)
                elif data["name"] == "audience":
                    entityArray.append(data)
                if 'week end' in utterence or 'weekend' in utterence:
                    if 'weekend' not in conditionMap:
                        data['name'] = 'weekend'
                        data['value'] = 'weekend'
                        entityArray.append(data)
                        conditionMap['weekend'] = 'weekend'
                else:
                    pass
        elif intent == "optionintent":
            for data in entMap:
                if data["name"] == 'ordinal':
                    data["name"] = 'option'
                    data['value'] = str(data['value'])
                    entityArray.append(data)
                elif data["name"] == 'number':
                    data["name"] = 'option'
                    data['value'] = str(data['value'])
                    entityArray.append(data)
        elif intent == "listpickupintent":
            for data in entMap:
                if data["name"] == 'pickup':
                    entityArray.append(data)
        elif intent == "feeinfointent":
            for data in entMap:
                if data["name"] == 'fee':
                    entityArray.append(data)
        elif intent == "reserve_searchintent":
            for data in entMap:
                if data["name"] == 'subject':
                    entityArray.append(data)
        elif intent == "libraryinfointent":
            condMap = {}
            for data in entMap:
                if data['name'] == 'libname':
                    entityArray.append(data)
                elif data['name'] == 'library':
                    data['name'] = 'libname'
                    entityArray.append(data)
                elif data['name'] == 'libinfofilter':
                    entityArray.append(data)
                elif data['name'] == 'currently':
                    condMap['currently'] = 'now'
                    entityArray.append(data)
                elif data["name"] == "time":
                    if 'from' in data['value']:
                        print("*********************************************************")
                        data['value'] = data['value'].replace("\'", "\"", -1)
                        datamap = json.loads(data['value'])
                        tempMap = {}
                        tempMap['name'] = 'from'
                        fromDate = datamap['from'].split("T")
                        tempMap['value'] = fromDate[0]
                        entityArray.append(tempMap)
                        tempMap = {}
                        tempMap['name'] = 'to'
                        todate = datamap['to'].split("T")
                        tempMap['value'] = todate[0]
                        entityArray.append(tempMap)
                    else:
                        date = data["value"].split("T")
                        print(type(data["value"]))
                        print(date)
                        data["name"] = 'hdate'
                        data["value"] = date[0]
                        if 'currently' in condMap:
                            pass
                        else:
                            entityArray.append(data)
                    conditionMap["hdate"] = data["value"]
                elif data["name"].lower() == 'date':
                    data["name"] = 'date'
                elif data["name"] == "hdate":
                    if "hdate" not in conditionMap:
                        entityArray.append(data)
                elif data['name'] == 'address' or data['name'] == 'contact' or data['name'] == 'details':
                    if data['name'] == 'address':
                        data['name'] = 'libinfofilter'
                        data['value'] = 'address'
                    elif data['name'] == 'contact':
                        data['name'] = 'libinfofilter'
                        data['value'] = 'contact'
                    elif data['name'] == 'details':
                        data['name'] = 'libinfofilter'
                        data['value'] = 'details'
                    entityArray.append(data)
            if 'week end' in utterence or 'weekend' in utterence:
                if 'weekend' not in conditionMap:
                    data['name'] = 'weekend'
                    data['value'] = 'weekend'
                    entityArray.append(data)
                    conditionMap['weekend'] = 'weekend'
        elif intent == "listintransitintent":
            data = {}
            data['name'] = 'inTransit'
            data['value'] = 'in transit'
            entityArray.append(data)
        elif intent == "switchpatronintent":
            for data in entMap:
                if data['name'] == 'person':
                    data['name'] = 'patronname'
                    entityArray.append(data)
        elif intent == "updateholdintent":
            conditionMap = {}
            for data in entMap:
                if data['name'] == 'holdFilter' or data['name'] == 'mtype':
                    if data['name'] == 'holdFilter':
                        conditionMap['holdFilter'] = True
                    entityArray.append(data)
                elif data["name"] == "WORK_OF_ART":
                    data["name"] = "stitle"
                    data["value"] = data["value"].lower().replace("search for a book", "").replace(
                        "search for the book", "").replace("serach for title", "").replace("search for a title",
                                                                                           "").replace(
                        "serach for the title", "")
                    if data["value"] != "":
                        if 'filterphrase' in conditionMap and conditionMap['WORK_OF_ART'] == conditionMap[
                            'filterphrase']:
                            pass
                        else:
                            entityArray.append(data)
                            conditionMap["stitle"] = data["value"]
                    print("Entity array after work of art is ", entityArray)
                elif data["name"] == "person":
                    if "stitle" in conditionMap:
                        if data["value"] != conditionMap["stitle"]:
                            data["name"] = "sauthor"
                            entityArray.append(data)
                    else:
                        data["name"] = "sauthor"
                        entityArray.append(data)
                elif data["name"] == "sBook":
                    if "stitle" not in conditionMap:
                        data["name"] = "stitle"
                        entityArray.append(data)
                        conditionMap["stitle"] = data["value"]
                elif data["name"] == "sbook":
                    if "stitle" not in conditionMap:
                        if len(data["value"]) <= 50:
                            data["name"] = "stitle"
                            entityArray.append(data)
                            conditionMap["stitle"] = data["value"]
                        else:
                            if "subject" not in conditionMap:
                                data["name"] = "subject"
                                data["name"] = data["name"].lower()
                                entityArray.append(data)
                                conditionMap["subject"] = data["value"]
                    else:
                        print(entityArray)
                        entityArray.pop(0)
                        conditionMap["stitle"] = data["value"].lower()
                        data["value"] = data["value"].lower()
                        data["name"] = "stitle"
                        entityArray.append(data)
                        print(entityArray)
                elif data["name"] == "time":
                    if 'from' in data['value']:
                        print("*********************************************************")
                        data['value'] = data['value'].replace("\'", "\"", -1)
                        datamap = json.loads(data['value'])
                        tempMap = {}
                        tempMap['name'] = 'from'
                        fromDate = datamap['from'].split("T")
                        tempMap['value'] = fromDate[0]
                        entityArray.append(tempMap)
                        tempMap = {}
                        tempMap['name'] = 'to'
                        todate = datamap['to'].split("T")
                        tempMap['value'] = todate[0]
                        entityArray.append(tempMap)
                    else:
                        date = data["value"].split("T")
                        print(type(data["value"]))
                        print(date)
                        data["name"] = 'hdate'
                        data["value"] = date[0]
                        if 'currently' in conditionMap:
                            pass
                        else:
                            entityArray.append(data)
                    conditionMap["hdate"] = data["value"]
            if 'holdFilter' not in conditionMap:
                if 'suspend' in utterence or 'deactivate' in utterence or 'disable' in utterence:
                    data = {}
                    data["name"] = 'holdFilter'
                    data["value"] = 'suspend'
                    entityArray.append(data)
                elif 'activate' in utterence or 'reactivate' in utterence:
                    data = {}
                    data["name"] = 'holdFilter'
                    data["value"] = 'activate'
                    entityArray.append(data)
        else:
            for data in entMap:
                entityArray.append(data)
        print(entityArray)
        return entityArray

    def entitySerializer(enityData):
        filterMap = []
        entityArray = []
        entityMap = {}
        for data in enityData:
            if data["entity"] == "PERSON":
                entData = data["value"]
                entArray = entData.split("||")
                if len(entArray) > 1:
                    for k in entArray:
                        entityMap["value"] = k
                        entityMap["name"] = "PERSON"
                        entityArray.append(entityMap)
                        entityMap = {}
                else:
                    entityMap["value"] = str(data["value"])
                    entityMap["name"] = data["entity"]
                    entityArray.append(entityMap)
                    entityMap = {}
            else:
                entityMap["value"] = str(data["value"])
                entityMap["name"] = data["entity"]
                if "synonym" in data:
                    entityMap["synonym"] = data["synonym"]
                entityArray.append(entityMap)
                entityMap = {}

        return entityArray

    @app.put("/model")
    @requires_auth(app, auth_token)
    async def load_model(request: Request):
        validate_request_body(request, "No path to model file defined in request_body.")

        model_path = request.json.get("model_file", None)
        model_server = request.json.get("model_server", None)
        remote_storage = request.json.get("remote_storage", None)
        if model_server:
            try:
                model_server = EndpointConfig.from_dict(model_server)
            except TypeError as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    400,
                    "BadRequest",
                    "Supplied 'model_server' is not valid. Error: {}".format(e),
                    {"parameter": "model_server", "in": "body"},
                )
        app.agent = await _load_agent(
            model_path, model_server, remote_storage, endpoints
        )

        logger.debug("Successfully loaded model '{}'.".format(model_path))
        return response.json(None, status=204)

    @app.delete("/model")
    @requires_auth(app, auth_token)
    async def unload_model(request: Request):
        model_file = app.agent.model_directory

        app.agent = Agent()

        logger.debug("Successfully unload model '{}'.".format(model_file))
        return response.json(None, status=204)

    @app.get("/domain")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def get_domain(request: Request):
        """Get current domain in yaml or json format."""

        accepts = request.headers.get("Accept", default="application/json")
        if accepts.endswith("json"):
            domain = app.agent.domain.as_dict()
            return response.json(domain)
        elif accepts.endswith("yml") or accepts.endswith("yaml"):
            domain_yaml = app.agent.domain.as_yaml()
            return response.text(
                domain_yaml, status=200, content_type="application/x-yml"
            )
        else:
            raise ErrorResponse(
                406,
                "NotAcceptable",
                "Invalid Accept header. Domain can be "
                "provided as "
                'json ("Accept: application/json") or'
                'yml ("Accept: application/x-yml"). '
                "Make sure you've set the appropriate Accept "
                "header.",
            )

    return app


def _get_output_channel(
        request: Request, tracker: Optional[DialogueStateTracker]
) -> OutputChannel:
    """Returns the `OutputChannel` which should be used for the bot's responses.
    Args:
        request: HTTP request whose query parameters can specify which `OutputChannel`
                 should be used.
        tracker: Tracker for the conversation. Used to get the latest input channel.
    Returns:
        `OutputChannel` which should be used to return the bot's responses to.
    """
    requested_output_channel = request.args.get(OUTPUT_CHANNEL_QUERY_KEY)

    if (
            requested_output_channel == USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL
            and tracker
    ):
        requested_output_channel = tracker.get_latest_input_channel()

    # Interactive training does not set `input_channels`, hence we have to be cautious
    registered_input_channels = getattr(request.app, "input_channels", None) or []
    matching_channels = [
        channel
        for channel in registered_input_channels
        if channel.name() == requested_output_channel
    ]

    # Check if matching channels can provide a valid output channel,
    # otherwise use `CollectingOutputChannel`
    return reduce(
        lambda output_channel_created_so_far, input_channel: (
                input_channel.get_output_channel() or output_channel_created_so_far
        ),
        matching_channels,
        CollectingOutputChannel(),
    )

    return app


def _get_output_channel(
        request: Request, tracker: Optional[DialogueStateTracker]
) -> OutputChannel:
    """Returns the `OutputChannel` which should be used for the bot's responses.
    Args:
        request: HTTP request whose query parameters can specify which `OutputChannel`
                 should be used.
        tracker: Tracker for the conversation. Used to get the latest input channel.
    Returns:
        `OutputChannel` which should be used to return the bot's responses to.
    """
    requested_output_channel = request.args.get(OUTPUT_CHANNEL_QUERY_KEY)

    if (
            requested_output_channel == USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL
            and tracker
    ):
        requested_output_channel = tracker.get_latest_input_channel()

    # Interactive training does not set `input_channels`, hence we have to be cautious
    registered_input_channels = getattr(request.app, "input_channels", None) or []
    matching_channels = [
        channel
        for channel in registered_input_channels
        if channel.name() == requested_output_channel
    ]

    # Check if matching channels can provide a valid output channel,
    # otherwise use `CollectingOutputChannel`
    return reduce(
        lambda output_channel_created_so_far, input_channel: (
                input_channel.get_output_channel() or output_channel_created_so_far
        ),
        matching_channels,
        CollectingOutputChannel(),
    )
