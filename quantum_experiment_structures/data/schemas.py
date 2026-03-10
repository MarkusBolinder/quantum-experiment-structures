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

CCS_GENERATOR_SETTINGS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {
        "range": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 2,
            "maxItems": 2,
        }
    },
    "type": "object",
    "properties": {
        "n_measurements_range": {"allOf": [{"$ref": "#/$defs/range"}, {"default": [3, 6]}]},
        "n_values_range": {"allOf": [{"$ref": "#/$defs/range"}, {"default": [2, 2]}]},
        "n_contexts_range": {"allOf": [{"$ref": "#/$defs/range"}, {"default": [2, 5]}]},
        "context_size_range": {"allOf": [{"$ref": "#/$defs/range"}, {"default": [2, 3]}]},
        "n_alternatives_range": {"allOf": [{"$ref": "#/$defs/range"}, {"default": [1, 3]}]},
        "enabling_relation_size_range": {"allOf": [{"$ref": "#/$defs/range"}, {"default": [1, 4]}]},
        "n_samples_per_causal_structure": {"type": "integer", "default": 1, "minimum": 1},
        "p_has_enabled": {"type": "number", "default": 0.6, "minimum": 0.0, "maximum": 1.0},
        "n_alternatives_mean": {"type": "number", "default": 1.2, "minimum": 0.0},
        "enabling_relation_size_mean": {"type": "number", "default": 1.3, "minimum": 0.0},
        "no_lexicographic_order": {"type": "boolean", "default": False},
        "output_dir": {"oneOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "batch_size": {"type": "integer", "default": 1, "minimum": 1},
        "n_scenarios": {"type": "integer", "default": 1, "minimum": 1},
        "seed": {"oneOf": [{"type": "integer"}, {"type": "null"}], "default": None},
    },
    "additionalProperties": False,
}
