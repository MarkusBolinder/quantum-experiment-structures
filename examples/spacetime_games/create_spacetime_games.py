import itertools
import json
from pathlib import Path

# Example 1: Bell experiment, perfect information.
histories1 = []
for a, b, ao, bo in itertools.product(["X", "Y"], ["W", "Z"], [0, 1], [0, 1]):
    histories1.append(
        {
            "z": f"z_{a}{b}_{ao}{bo}",
            "h": [
                {"i": "I_A", "a": a},
                {"i": "I_Aout", "a": str(ao)},
                {"i": "I_B", "a": b},
                {"i": "I_Bout", "a": str(bo)},
            ],
            "p": ["I_A", "I_Aout", "I_B", "I_Bout"],
            "l": True,
            "u": [
                {"p": "Alice", "v": 0},
                {"p": "Bob", "v": 0},
                {"p": "Alfred", "v": 0},
            ],
        }
    )

game1 = {
    "ps": ["Alice", "Bob", "Alfred"],
    "as": ["X", "Y", "W", "Z", "0", "1"],
    "ns": [
        {"n": "nA", "p": "Alice", "a": ["X", "Y"], "i": ["I_A"]},
        {"n": "nB", "p": "Bob", "a": ["W", "Z"], "i": ["I_B"]},
        {"n": "nAout", "p": "Alfred", "a": ["0", "1"], "i": ["I_Aout"]},
        {"n": "nBout", "p": "Alfred", "a": ["0", "1"], "i": ["I_Bout"]},
    ],
    "es": [
        {"f": "nA", "t": "nAout", "a": "X"},
        {"f": "nA", "t": "nAout", "a": "Y"},
        {"f": "nB", "t": "nBout", "a": "W"},
        {"f": "nB", "t": "nBout", "a": "Z"},
    ],
    "is": [
        {"i": "I_A", "n": ["nA"], "p": "Alice", "a": ["X", "Y"]},
        {"i": "I_B", "n": ["nB"], "p": "Bob", "a": ["W", "Z"]},
        {"i": "I_Aout", "n": ["nAout"], "p": "Alfred", "a": ["0", "1"]},
        {"i": "I_Bout", "n": ["nBout"], "p": "Alfred", "a": ["0", "1"]},
    ],
    "z": histories1,
    "u": [
        {"p": "Alice", "u": [{"z": h["z"], "v": 0} for h in histories1]},
        {"p": "Bob", "u": [{"z": h["z"], "v": 0} for h in histories1]},
        {"p": "Alfred", "u": [{"z": h["z"], "v": 0} for h in histories1]},
    ],
}

# Example 2: cyclic rank 3, imperfect information.
histories2 = []
for ctx, infos in [
    ("XY", ("I_X", "I_Y")),
    ("XZ", ("I_X", "I_Z")),
    ("YZ", ("I_Y", "I_Z")),
]:
    for o1, o2 in itertools.product([0, 1], [0, 1]):
        histories2.append(
            {
                "z": f"z_{ctx}_{o1}{o2}",
                "h": [
                    {"i": "I_B", "a": ctx},
                    {"i": infos[0], "a": str(o1)},
                    {"i": infos[1], "a": str(o2)},
                ],
                "p": ["I_B", infos[0], infos[1]],
                "l": True,
                "u": [
                    {"p": "Bob", "v": 0},
                    {"p": "Alfred", "v": 0},
                ],
            }
        )

game2 = {
    "ps": ["Bob", "Alfred"],
    "as": ["XY", "XZ", "YZ", "0", "1"],
    "ns": [
        {"n": "nB", "p": "Bob", "a": ["XY", "XZ", "YZ"], "i": ["I_B"]},
        {"n": "nXY_X", "p": "Alfred", "a": ["0", "1"], "i": ["I_X"]},
        {"n": "nXY_Y", "p": "Alfred", "a": ["0", "1"], "i": ["I_Y"]},
        {"n": "nXZ_X", "p": "Alfred", "a": ["0", "1"], "i": ["I_X"]},
        {"n": "nXZ_Z", "p": "Alfred", "a": ["0", "1"], "i": ["I_Z"]},
        {"n": "nYZ_Y", "p": "Alfred", "a": ["0", "1"], "i": ["I_Y"]},
        {"n": "nYZ_Z", "p": "Alfred", "a": ["0", "1"], "i": ["I_Z"]},
    ],
    "es": [
        {"f": "nB", "t": "nXY_X", "a": "XY"},
        {"f": "nXY_X", "t": "nXY_Y", "a": "0"},
        {"f": "nXY_X", "t": "nXY_Y", "a": "1"},
        {"f": "nB", "t": "nXZ_X", "a": "XZ"},
        {"f": "nXZ_X", "t": "nXZ_Z", "a": "0"},
        {"f": "nXZ_X", "t": "nXZ_Z", "a": "1"},
        {"f": "nB", "t": "nYZ_Y", "a": "YZ"},
        {"f": "nYZ_Y", "t": "nYZ_Z", "a": "0"},
        {"f": "nYZ_Y", "t": "nYZ_Z", "a": "1"},
    ],
    "is": [
        {"i": "I_B", "n": ["nB"], "p": "Bob", "a": ["XY", "XZ", "YZ"]},
        {"i": "I_X", "n": ["nXY_X", "nXZ_X"], "p": "Alfred", "a": ["0", "1"]},
        {"i": "I_Y", "n": ["nXY_Y", "nYZ_Y"], "p": "Alfred", "a": ["0", "1"]},
        {"i": "I_Z", "n": ["nXZ_Z", "nYZ_Z"], "p": "Alfred", "a": ["0", "1"]},
    ],
    "z": histories2,
    "u": [
        {"p": "Bob", "u": [{"z": h["z"], "v": 0} for h in histories2]},
        {"p": "Alfred", "u": [{"z": h["z"], "v": 0} for h in histories2]},
    ],
}

# Example 3: causal bridge, imperfect information via shared outcome info set.
histories3 = [
    {
        "z": "z_X0",
        "h": [{"i": "I_A", "a": "X"}, {"i": "I_O", "a": "0"}],
        "p": ["I_A", "I_O"],
        "l": True,
        "u": [{"p": "Alice", "v": 0}, {"p": "Bob", "v": 0}, {"p": "Nature", "v": 0}],
    },
    {
        "z": "z_X1",
        "h": [{"i": "I_A", "a": "X"}, {"i": "I_O", "a": "1"}],
        "p": ["I_A", "I_O"],
        "l": True,
        "u": [{"p": "Alice", "v": 0}, {"p": "Bob", "v": 0}, {"p": "Nature", "v": 0}],
    },
    {
        "z": "z_Y1",
        "h": [{"i": "I_A", "a": "Y"}, {"i": "I_O", "a": "1"}],
        "p": ["I_A", "I_O"],
        "l": True,
        "u": [{"p": "Alice", "v": 0}, {"p": "Bob", "v": 0}, {"p": "Nature", "v": 0}],
    },
    {
        "z": "z_Y0_b0",
        "h": [{"i": "I_A", "a": "Y"}, {"i": "I_O", "a": "0"}, {"i": "I_B", "a": "b0"}],
        "p": ["I_A", "I_O", "I_B"],
        "l": True,
        "u": [{"p": "Alice", "v": 0}, {"p": "Bob", "v": 0}, {"p": "Nature", "v": 0}],
    },
    {
        "z": "z_Y0_b1",
        "h": [{"i": "I_A", "a": "Y"}, {"i": "I_O", "a": "0"}, {"i": "I_B", "a": "b1"}],
        "p": ["I_A", "I_O", "I_B"],
        "l": True,
        "u": [{"p": "Alice", "v": 0}, {"p": "Bob", "v": 0}, {"p": "Nature", "v": 0}],
    },
]

game3 = {
    "ps": ["Alice", "Bob", "Nature"],
    "as": ["X", "Y", "0", "1", "b0", "b1"],
    "ns": [
        {"n": "nA", "p": "Alice", "a": ["X", "Y"], "i": ["I_A"]},
        {"n": "nAX", "p": "Nature", "a": ["0", "1"], "i": ["I_O"]},
        {"n": "nAY", "p": "Nature", "a": ["0", "1"], "i": ["I_O"]},
        {"n": "nB", "p": "Bob", "a": ["b0", "b1"], "i": ["I_B"]},
    ],
    "es": [
        {"f": "nA", "t": "nAX", "a": "X"},
        {"f": "nA", "t": "nAY", "a": "Y"},
        {"f": "nAY", "t": "nB", "a": "0"},
    ],
    "is": [
        {"i": "I_A", "n": ["nA"], "p": "Alice", "a": ["X", "Y"]},
        {"i": "I_O", "n": ["nAX", "nAY"], "p": "Nature", "a": ["0", "1"]},
        {"i": "I_B", "n": ["nB"], "p": "Bob", "a": ["b0", "b1"]},
    ],
    "z": histories3,
    "u": [
        {"p": "Alice", "u": [{"z": h["z"], "v": 0} for h in histories3]},
        {"p": "Bob", "u": [{"z": h["z"], "v": 0} for h in histories3]},
        {"p": "Nature", "u": [{"z": h["z"], "v": 0} for h in histories3]},
    ],
}

# Example 4: Bell experiment as cyclic rank 4 contextuality.
branches = [
    ("XY", ("I_X", "I_Y")),
    ("YW", ("I_Y", "I_W")),
    ("WZ", ("I_W", "I_Z")),
    ("ZX", ("I_Z", "I_X")),
]
histories4 = []
for ctx, infos in branches:
    for o1, o2 in itertools.product([0, 1], [0, 1]):
        histories4.append(
            {
                "z": f"z_{ctx}_{o1}{o2}",
                "h": [
                    {"i": "I_A", "a": ctx},
                    {"i": infos[0], "a": str(o1)},
                    {"i": infos[1], "a": str(o2)},
                ],
                "p": ["I_A", infos[0], infos[1]],
                "l": True,
                "u": [{"p": "Observer", "v": 0}, {"p": "Nature", "v": 0}],
            }
        )

game4 = {
    "ps": ["Observer", "Nature"],
    "as": ["XY", "YW", "WZ", "ZX", "0", "1"],
    "ns": [
        {"n": "nA", "p": "Observer", "a": ["XY", "YW", "WZ", "ZX"], "i": ["I_A"]},
        {"n": "nXY_X", "p": "Nature", "a": ["0", "1"], "i": ["I_X"]},
        {"n": "nXY_Y", "p": "Nature", "a": ["0", "1"], "i": ["I_Y"]},
        {"n": "nYW_Y", "p": "Nature", "a": ["0", "1"], "i": ["I_Y"]},
        {"n": "nYW_W", "p": "Nature", "a": ["0", "1"], "i": ["I_W"]},
        {"n": "nWZ_W", "p": "Nature", "a": ["0", "1"], "i": ["I_W"]},
        {"n": "nWZ_Z", "p": "Nature", "a": ["0", "1"], "i": ["I_Z"]},
        {"n": "nZX_Z", "p": "Nature", "a": ["0", "1"], "i": ["I_Z"]},
        {"n": "nZX_X", "p": "Nature", "a": ["0", "1"], "i": ["I_X"]},
    ],
    "es": [
        {"f": "nA", "t": "nXY_X", "a": "XY"},
        {"f": "nXY_X", "t": "nXY_Y", "a": "0"},
        {"f": "nXY_X", "t": "nXY_Y", "a": "1"},
        {"f": "nA", "t": "nYW_Y", "a": "YW"},
        {"f": "nYW_Y", "t": "nYW_W", "a": "0"},
        {"f": "nYW_Y", "t": "nYW_W", "a": "1"},
        {"f": "nA", "t": "nWZ_W", "a": "WZ"},
        {"f": "nWZ_W", "t": "nWZ_Z", "a": "0"},
        {"f": "nWZ_W", "t": "nWZ_Z", "a": "1"},
        {"f": "nA", "t": "nZX_Z", "a": "ZX"},
        {"f": "nZX_Z", "t": "nZX_X", "a": "0"},
        {"f": "nZX_Z", "t": "nZX_X", "a": "1"},
    ],
    "is": [
        {"i": "I_A", "n": ["nA"], "p": "Observer", "a": ["XY", "YW", "WZ", "ZX"]},
        {"i": "I_X", "n": ["nXY_X", "nZX_X"], "p": "Nature", "a": ["0", "1"]},
        {"i": "I_Y", "n": ["nXY_Y", "nYW_Y"], "p": "Nature", "a": ["0", "1"]},
        {"i": "I_W", "n": ["nYW_W", "nWZ_W"], "p": "Nature", "a": ["0", "1"]},
        {"i": "I_Z", "n": ["nWZ_Z", "nZX_Z"], "p": "Nature", "a": ["0", "1"]},
    ],
    "z": histories4,
    "u": [
        {"p": "Observer", "u": [{"z": h["z"], "v": 0} for h in histories4]},
        {"p": "Nature", "u": [{"z": h["z"], "v": 0} for h in histories4]},
    ],
}

payload = {
    "bell_perfect_information": game1,
    "cyclic_rank_3_imperfect_information": game2,
    "causal_bridge_imperfect_information": game3,
    "cyclic_rank_4_imperfect_information": game4,
}

out = Path(".")
out.write_text(json.dumps(payload, indent=4))
print(f"Wrote {out} ({out.stat().st_size} bytes)")
