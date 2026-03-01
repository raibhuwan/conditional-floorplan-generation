# Merged class IDs
BG = 0
KITCHEN = 1
LIVING = 2
BEDROOM = 3
BATHROOM = 4
HALLWAY = 5
DINING = 6
UTILITY = 7
WALL = 8

# Rules: CubiCasa "Space X" token -> merged class id
TOKEN_TO_MERGED = {
    # main rooms
    "Kitchen": KITCHEN,
    "LivingRoom": LIVING,
    "Lounge": LIVING,
    "Bedroom": BEDROOM,

    # dining
    "DiningRoom": DINING,
    "Dining": DINING,

    # bathroom family
    "Bath": BATHROOM,
    "Bathroom": BATHROOM,
    "WC": BATHROOM,
    "Toilet": BATHROOM,
    "Sauna": BATHROOM,
    "Shower": BATHROOM,

    # circulation
    "Hallway": HALLWAY,
    "Hall": HALLWAY,
    "Entry": HALLWAY,
    "Foyer": HALLWAY,
    "Corridor": HALLWAY,

    # utility/storage family
    "Storage": UTILITY,
    "Closet": UTILITY,
    "WalkIn": UTILITY,
    "Laundry": UTILITY,
    "Utility": UTILITY,
    "Pantry": UTILITY,

    # structure
    "Wall": WALL,

        # ---- added from unknown tokens ----

    # room-like spaces
    "Office": LIVING,
    "Den": LIVING,
    "Room": LIVING,

    # entrance / circulation
    "DraughtLobby": HALLWAY,

    # utility-like spaces
    "Garage": UTILITY,
    "CarPort": UTILITY,
    "TechnicalRoom": UTILITY,
    "DressingRoom": UTILITY,

    # outdoor (we treat as background / ignore)
    "Outdoor": BG,

    # structure/elements we choose to ignore (map to BG so they don't become rooms)
    "Door": BG,
    "Window": BG,
    "Railing": BG,
    "Column": BG,
    "Threshold": BG,
    "Glass": BG,

    # drawings / polygons / UI artifacts (ignore)
    "BoundaryPolygon": BG,
    "InnerPolygon": BG,
    "InnerPolygonTop": BG,
    "InnerPolygonBottom": BG,
    "InnerPolygonLeft": BG,
    "InnerPolygonRight": BG,
    "OverlayPolygon": BG,
    "PanelArea": BG,
    "DimensionMark": BG,
    "Direction": BG,
    "Faucet": BG,
    "FireBox": BG,
    "Flight": BG,
    "RoundedWinding": BG,
    "SelectionControls": BG,
    "Undefined": BG,
    "UserDefined": BG,
    "WalkinLine": BG,
    "copyPasteControl": BG,
    "removeControl": BG,
    "translateControl": BG,

    # extra room-like tokens (map to utility or living)
    "Alcove": UTILITY,
    "Attic": UTILITY,
    "Basement": UTILITY,
    "Elevated": HALLWAY,        # or UTILITY; hallway is fine
    "Elevator": HALLWAY,        # circulation element
    "Landing": HALLWAY,
    "Library": LIVING,
    "RecreationRoom": LIVING,
    "SwimmingPool": UTILITY,    # or BG if you prefer

    # stairs / geometry (ignore for now)
    "Stairs": BG,
    "Winding": BG,
}