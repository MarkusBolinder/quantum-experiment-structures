"""Statically stored schemas."""

CCS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "$defs": {
        "enabled_by": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["m", "v"],
                    "properties": {"m": {"type": "string"}, "v": {"type": "integer"}},
                    "additionalProperties": False,
                },
                "uniqueItems": True,
                "minItems": 1,
            },
            "uniqueItems": True,
        },
        "contexts": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
                "minItems": 1,
            },
            "uniqueItems": True,
        },
        "cover": {
            "allOf": [
                {"$ref": "#/$defs/contexts"},
                {"minItems": 1},  # force the cover to have at least one context
            ]
        },
    },
    "required": ["ms", "c"],
    "properties": {
        "ms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["m", "e", "o"],
                "properties": {
                    "m": {"type": "string"},
                    "e": {"$ref": "#/$defs/enabled_by"},
                    "o": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["v"],  # the leaf property can be calculated later
                            "properties": {"v": {"type": "integer"}, "l": {"type": "boolean"}},
                            "additionalProperties": False,
                        },
                        "uniqueItems": True,
                        "minItems": 1,
                    },
                    # no 'minItems'-requirement here because it can be calculated from the rest
                    "c": {"$ref": "#/$defs/contexts"},
                },
                "additionalProperties": False,
            },
            "uniqueItems": True,
            "minItems": 1,
        },
        "c": {"$ref": "#/$defs/cover"},
        "n": {"type": "array", "items": {"type": "string"}},
        "h": {
            "type": "object",
            "required": ["ms", "o", "e", "c"],
            "properties": {
                "ms": {"type": "string"},
                "o": {"type": "string"},
                "e": {"type": "string"},
                "c": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "additionalProperties": False,
    },
    "additionalProperties": False,
}
