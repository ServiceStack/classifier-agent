""" Options:
Date: 2025-09-11 01:12:44
Version: 8.81
Tip: To override a DTO option, remove "#" prefix before updating
BaseUrl: https://amd.raptor-elver.ts.net

#GlobalNamespace: 
#AddServiceStackTypes: True
#AddResponseStatus: False
#AddImplicitVersion: 
#AddDescriptionAsComments: True
IncludeTypes: GetArtifactClassificationTasks.*,CompleteArtifactClassificationTask.*,AgentData.*,ArtifactRef
#ExcludeTypes: 
#DefaultImports: datetime,decimal,marshmallow.fields:*,servicestack:*,typing:*,dataclasses:dataclass/field,dataclasses_json:dataclass_json/LetterCase/Undefined/config,enum:Enum/IntEnum
#DataClass: 
#DataClassJson: 
"""

import datetime
import decimal
from marshmallow.fields import *
from servicestack import *
from typing import *
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, LetterCase, Undefined, config
from enum import Enum, IntEnum


class AssetType(str, Enum):
    IMAGE = 'Image'
    VIDEO = 'Video'
    AUDIO = 'Audio'
    ANIMATION = 'Animation'
    TEXT = 'Text'
    BINARY = 'Binary'


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class ObjectDetection:
    model: Optional[str] = None
    class_: Optional[str] = field(metadata=config(field_name='class'), default=None)
    score: float = 0.0
    box: List[int] = field(default_factory=list)


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class Ratings:
    predicted_rating: Optional[str] = None
    confidence: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class ArtifactRef:
    id: int = 0
    type: Optional[AssetType] = None
    url: Optional[str] = None
    length: int = 0
    device_id: Optional[str] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class AgentDataResponse:
    categories: List[str] = field(default_factory=list)
    response_status: Optional[ResponseStatus] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class AgentData(IReturn[AgentDataResponse], IGet):
    # @Validate(Validator="NotEmpty")
    device_id: Optional[str] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class GetArtifactClassificationTasks(IReturn[QueryResponse[ArtifactRef]], IGet):
    # @Validate(Validator="NotEmpty")
    device_id: Optional[str] = None

    # @Validate(Validator="NotEmpty")
    types: List[AssetType] = field(default_factory=list)

    take: Optional[int] = None
    wait_for: Optional[int] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class CompleteArtifactClassificationTask(IReturn[EmptyResponse], IPost):
    # @Validate(Validator="NotEmpty")
    device_id: Optional[str] = None

    artifact_id: int = 0
    tags: Optional[Dict[str, float]] = None
    categories: Optional[Dict[str, float]] = None
    objects: Optional[List[ObjectDetection]] = None
    ratings: Optional[Ratings] = None
    phash: Optional[str] = None
    color: Optional[str] = None
    error: Optional[ResponseStatus] = None

