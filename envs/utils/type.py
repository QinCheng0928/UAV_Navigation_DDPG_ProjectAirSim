class ActionType:
    
    NORTH = 0   # +X
    EAST  = 1   # +Y
    DOWN  = 2   # +Z
    SOUTH = 3   # -X
    WEST  = 4   # -Y
    UP    = 5   # -Z
    BRAKE  = 6   # brake

    NUM2NAME = {
        NORTH: "North (+X)",
        EAST: "East (+Y)",
        DOWN: "Down (+Z)",
        SOUTH: "South (-X)",
        WEST: "West (-Y)",
        UP: "Up (-Z)",
        BRAKE: "Brake"
    }