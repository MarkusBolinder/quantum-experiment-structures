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
            },
        },
        "cover": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
    "required": ["ms", "c"],
    "properties": {
        "ms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["m", "e", "o", "c"],
                "properties": {
                    "m": {"type": "string"},
                    "e": {"$ref": "#/$defs/enabled_by"},
                    "o": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["v", "l"],
                            "properties": {"v": {"type": "integer"}, "l": {"type": "boolean"}},
                            "additionalProperties": False,
                        },
                    },
                    "c": {"$ref": "#/$defs/cover"},
                },
                "additionalProperties": False,
            },
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
