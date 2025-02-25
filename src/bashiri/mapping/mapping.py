import abc
import json
import os
from typing import Optional, Dict, List, Tuple, Generator

from sflkit.config import hash_identifier
from sflkit.events.mapping import EventMapping
from sflkitlib.events import EventType
from sflkitlib.events.event import (
    Event,
    LineEvent,
    BranchEvent,
    FunctionEnterEvent,
    FunctionExitEvent,
    FunctionErrorEvent,
    DefEvent,
    UseEvent,
    ConditionEvent,
    LoopBeginEvent,
    LoopEndEvent,
    LoopHitEvent,
    LenEvent,
)

from bashiri.mapping.patch import PatchTranslator


def identifier(target_path: os.PathLike):
    return hash_identifier(target_path)


class Mapping(abc.ABC):
    @abc.abstractmethod
    def map(self, event: Event):
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Generator[Tuple[int, Event], None, None]:
        pass

    @abc.abstractmethod
    def __contains__(self, item) -> bool:
        pass

    def __getitem__(self, key) -> Event:
        if not isinstance(key, Event):
            raise TypeError("key must be an instance of Event")
        return self.map(key)

    @abc.abstractmethod
    def dump(self, path: os.PathLike, indent: Optional[int] = None):
        pass

    def get_translation(self) -> Dict[int, int]:
        translation = dict()
        for event_id, event in self:
            mapped = self.map(event)
            if mapped:
                translation[event_id] = self.map(event).event_id
        return translation


class IdMapping(Mapping):
    def __init__(
        self,
        event_mapping: EventMapping,
        mapping: Optional[Dict[int, Optional[int]]] = None,
    ):
        self.event_mapping = event_mapping
        self.mapping = mapping

    def map(self, event: Event):
        if event.event_id in self.mapping:
            return self.event_mapping.get(self.mapping[event.event_id])
        return None

    def dump(self, path: os.PathLike, indent: Optional[int] = None):
        with open(path, "w") as f:
            json.dump(self.mapping, f, indent=indent)

    def load(self, path: os.PathLike):
        with open(path, "r") as f:
            self.mapping = json.load(f)
        return self

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Generator[Tuple[int, Event], None, None]:
        for event_id in self.mapping:
            yield event_id, self.event_mapping.get(self.mapping[event_id])

    def __contains__(self, item) -> bool:
        if isinstance(item, Event):
            return item.event_id in self.mapping
        elif isinstance(item, int):
            return item in self.mapping
        else:
            raise TypeError("item must be an instance of Event or int")


class ConcreteMapping(Mapping):
    def __init__(self, mapping: Optional[Dict[Event, Optional[Event]]] = None):
        self.mapping = mapping or dict()

    def map(self, event: Event):
        return self.mapping.get(event, None)

    def dump(self, path: os.PathLike, indent: Optional[int] = None):
        id_mapping = dict()
        for event, mapped in self.mapping.items():
            id_mapping[event.event_id] = mapped.event_id if mapped else None
        with open(path, "w") as f:
            json.dump(id_mapping, f, indent=indent)

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Generator[Tuple[int, Event], None, None]:
        for event in self.mapping:
            yield event.event_id, self.mapping[event]

    def __contains__(self, item) -> bool:
        if isinstance(item, Event):
            return item in self.mapping
        elif isinstance(item, int):
            for event in self.mapping:
                if event.event_id == item:
                    return True
            return False
        else:
            raise TypeError("item must be an instance of Event or int")


class MappingCreator:
    def __init__(self, origin: EventMapping):
        self.origin = origin
        self.location_map: Dict[str, Dict[int, Dict[EventType, List[Event]]]] = dict()
        self.rebuild()

    def rebuild(self):
        for event_id in self.origin:
            event = self.origin.get(event_id)
            if event:
                file = event.file
                if file not in self.location_map:
                    self.location_map[file] = dict()
                line = event.line
                if line not in self.location_map[file]:
                    self.location_map[file][line] = dict()
                event_type = event.event_type
                if event_type not in self.location_map[file][line]:
                    self.location_map[file][line][event_type] = list()
                self.location_map[file][line][event_type].append(event)

    def get_possible_events(
        self, event: Event, translator: PatchTranslator
    ) -> Optional[List[Event]]:
        file, lines = translator.translate((event.file, event.line))
        event_type = event.event_type
        if file in self.location_map:
            possible_events = list()
            for line in lines:
                if line in self.location_map[file]:
                    if event_type in self.location_map[file][line]:
                        possible_events += self.location_map[file][line][event_type]
            return possible_events or None
        return None

    def map_generic(self, event: Event, translator: PatchTranslator) -> Optional[Event]:
        events = self.get_possible_events(event, translator)
        if events:
            return events[0]
        else:
            return None

    def map_line(self, event: LineEvent, translator: PatchTranslator):
        return self.map_generic(event, translator)

    def map_branch(self, event: BranchEvent, translator: PatchTranslator):
        events = self.get_possible_events(event, translator)
        if events:
            for candidate in events:
                candidate: BranchEvent
                if event.then_id > event.else_id:
                    if candidate.then_id > candidate.else_id:
                        return candidate
                elif event.then_id < event.else_id:
                    if candidate.then_id < candidate.else_id:
                        return candidate
        else:
            return None

    def map_function_enter(
        self, event: FunctionEnterEvent, translator: PatchTranslator
    ):
        return self.map_generic(event, translator)

    def map_function_error(
        self, event: FunctionErrorEvent, translator: PatchTranslator
    ):
        return self.map_generic(event, translator)

    def map_function_exit(self, event: FunctionExitEvent, translator: PatchTranslator):
        return self.map_generic(event, translator)

    def map_def(self, event: DefEvent, translator: PatchTranslator):
        events = self.get_possible_events(event, translator)
        if events:
            for candidate in events:
                candidate: DefEvent
                if event.var == candidate.var:
                    return candidate
        else:
            return None

    def map_use(self, event: UseEvent, translator: PatchTranslator):
        events = self.get_possible_events(event, translator)
        if events:
            for candidate in events:
                candidate: UseEvent
                if event.var == candidate.var:
                    return candidate
        else:
            return None

    def map_condition(self, event: ConditionEvent, translator: PatchTranslator):
        events = self.get_possible_events(event, translator)
        if events:
            for candidate in events:
                candidate: ConditionEvent
                if event.condition == candidate.condition:
                    return candidate
        else:
            return None

    def map_loop_begin(self, event: LoopBeginEvent, translator: PatchTranslator):
        return self.map_generic(event, translator)

    def map_loop_hit(self, event: LoopHitEvent, translator: PatchTranslator):
        return self.map_generic(event, translator)

    def map_loop_end(self, event: LoopEndEvent, translator: PatchTranslator):
        return self.map_generic(event, translator)

    def map_len_event(self, event: LenEvent, translator: PatchTranslator):
        events = self.get_possible_events(event, translator)
        if events:
            for candidate in events:
                candidate: LenEvent
                if event.var == candidate.var:
                    return candidate
        else:
            return None

    def create(self, target: EventMapping, translator: PatchTranslator) -> Mapping:
        mapping = dict()
        for event_id in target:
            event = target.get(event_id)
            if event:
                match event.event_type:
                    case EventType.LINE:
                        mapped = self.map_line(event, translator)
                    case EventType.BRANCH:
                        mapped = self.map_branch(event, translator)
                    case EventType.FUNCTION_ENTER:
                        mapped = self.map_function_enter(event, translator)
                    case EventType.FUNCTION_EXIT:
                        mapped = self.map_function_exit(event, translator)
                    case EventType.FUNCTION_ERROR:
                        mapped = self.map_function_error(event, translator)
                    case EventType.DEF:
                        mapped = self.map_def(event, translator)
                    case EventType.USE:
                        mapped = self.map_use(event, translator)
                    case EventType.CONDITION:
                        mapped = self.map_condition(event, translator)
                    case EventType.LOOP_BEGIN:
                        mapped = self.map_loop_begin(event, translator)
                    case EventType.LOOP_HIT:
                        mapped = self.map_loop_hit(event, translator)
                    case EventType.LOOP_END:
                        mapped = self.map_loop_end(event, translator)
                    case EventType.LEN:
                        mapped = self.map_len_event(event, translator)
                    case _:
                        mapped = None
                if mapped:
                    mapping[event] = mapped
        return ConcreteMapping(mapping)
