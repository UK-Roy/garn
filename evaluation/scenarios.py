"""
Test scenario configurations matching paper Table I.
S1: 6 individuals, 8 obstacles, 1 static group (2), 1 dynamic group (3)
S2: 6 individuals, 8 obstacles, 2 static groups (2,3), 2 dynamic groups (3,4)
"""

SCENARIOS = {
    's1': {
        'n_individuals': 6,
        'n_obstacles': 8,
        'static_groups': [
            {'members': 2},
        ],
        'dynamic_groups': [
            {'members': 3},
        ],
    },
    's2': {
        'n_individuals': 6,
        'n_obstacles': 8,
        'static_groups': [
            {'members': 2},
            {'members': 3},
        ],
        'dynamic_groups': [
            {'members': 3},
            {'members': 4},
        ],
    },
}
