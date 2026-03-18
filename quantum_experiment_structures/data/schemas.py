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

SPACETIME_GAME_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "$defs": {
        "id": {"type": "string"},
        "string_array": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
            "minItems": 1,
        },
        "player": {"type": "string"},
        "action": {"type": "string"},
        "node": {
            "type": "object",
            "required": ["n", "ps", "cn"],
            "properties": {
                "n": {"type": "string"},
                "ps": {  # parents of this node
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["p", "a"],  # parent and action performed at parent
                        "properties": {"p": {"type": "string"}, "a": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    "uniqueItems": True,
                },
                "cn": {  # children of this node
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["c", "a"],  # child and action needed to get there
                        "properties": {"c": {"type": "string"}, "a": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    "uniqueItems": True,
                },
            },
            "additionalProperties": False,
        },
        "history_or_strategy": {
            "type": "array",
            "items": {"$ref": "#/$defs/assignment"},
            "uniqueItems": True,
            "minItems": 1,
        },
        "info_set": {"type": "string"},
        "assignment": {
            "type": "object",
            "required": ["i", "a"],
            "properties": {"i": {"$ref": "#/$defs/info_set"}, "a": {"$ref": "#/$defs/action"}},
            "additionalProperties": False,
        },
        "payoff": {
            "type": "object",
            "required": ["p", "v"],
            "properties": {"p": {"$ref": "#/$defs/player"}, "v": {"type": "number"}},
            "additionalProperties": False,
        },
    },
    "required": ["ps", "as", "is"],  # histories and strategies can be inferred
    "properties": {
        "ps": {"$ref": "#/$defs/string_array"},  # players
        "as": {"$ref": "#/$defs/string_array"},  # actions
        "is": {                                  # information sets
            "type": "array",
            "items": {
                "type": "object",
                # histories and strategies can be inferred from the rest
                # however, when inferring histories, there is no way to know payoffs,
                # so they will just be put to zero as placeholders
                "required": ["i", "ns", "p", "a"],
                "properties": {
                    "i": {"$ref": "#/$defs/info_set"},
                    "ns": {  # all nodes in this information set
                        "type": "array",
                        "items": {"$ref": "#/$defs/node"},
                        "uniqueItems": True,
                        "minItems": 1,
                    },
                    "p": {"$ref": "#/$defs/player"},  # player associated
                    "a": {"$ref": "#/$defs/string_array"},  # actions playable
                },
                "additionalProperties": False,
            },
            "uniqueItems": True,
            "minItems": 1,
        },
        "z": {                                   # histories
            "type": "array",
            "items": {
                "type": "object",
                "required": ["z", "h", "u"],
                "properties": {
                    "z": {"$ref": "#/$defs/id"},
                    "h": {"$ref": "#/$defs/history_or_strategy"},
                    "s": {"$ref": "#/$defs/string_array"},  # the information sets played in history
                    "u": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/payoff"},
                        "uniqueItems": True,
                        "minItems": 1,
                    },
                },
                "additionalProperties": False,
            },
            "uniqueItems": True,
            "minItems": 1,
        },
        "s": {                                   # strategies
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "p": {"$ref": "#/$defs/player"},
                    "s": {  # all strategies for this player
                        "type": "array",
                        "items": {"$ref": "#/$defs/history_or_strategy"},
                        "uniqueItems": True,
                        "minItems": 1,
                    },
                    "additionalProperties": False,
                },
            },
            "uniqueItems": True,
            "minItems": 1,
        },
        "n": {"type": "array", "items": {"type": "string"}},
        "h": {
            "type": "object",
            "required": ["ps", "ns", "es", "is", "z", "u", "s"],
            "properties": {
                "ps": {"type": "string"},
                "ns": {"type": "string"},
                "es": {"type": "string"},
                "is": {"type": "string"},
                "z": {"type": "string"},
                "u": {"type": "string"},
                "s": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}
