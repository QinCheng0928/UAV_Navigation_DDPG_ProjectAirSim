class ActionType:
    
    NORTH = 0   # +X
    EAST  = 1   # +Y
    DOWN  = 2   # +Z
    SOUTH = 3   # -X
    WEST  = 4   # -Y
    UP    = 5   # -Z
    NONE  = 6   # hover / no movement

    NUM2NAME = {
        NORTH: "North (+X)",
        EAST: "East (+Y)",
        DOWN: "Down (+Z)",
        SOUTH: "South (-X)",
        WEST: "West (-Y)",
        UP: "Up (-Z)",
        NONE: "None"
    }