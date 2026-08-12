# Define the nine merged semantic class IDs used throughout the project.
# Class 0 represents background, classes 1-7 represent room categories,
# and class 8 represents wall/structural regions.
BG = 0
KITCHEN = 1
LIVING = 2
BEDROOM = 3
BATHROOM = 4
HALLWAY = 5
DINING = 6
UTILITY = 7
WALL = 8

# Map CubiCasa source annotation tokens to the reduced nine-class
# semantic representation used for model training and evaluation.
TOKEN_TO_MERGED = {
    # Map the principal room categories directly to their merged classes.
    # main rooms
    "Kitchen": KITCHEN,
    "LivingRoom": LIVING,
    "Lounge": LIVING,
    "Bedroom": BEDROOM,

    # Merge alternative dining labels into a single dining-room class.
    # dining
    "DiningRoom": DINING,
    "Dining": DINING,

    # Merge bathroom-related source labels into one bathroom class.
    # bathroom family
    "Bath": BATHROOM,
    "Bathroom": BATHROOM,
    "WC": BATHROOM,
    "Toilet": BATHROOM,
    "Sauna": BATHROOM,
    "Shower": BATHROOM,

    # Merge circulation-related labels into the hallway class.
    # circulation
    "Hallway": HALLWAY,
    "Hall": HALLWAY,
    "Entry": HALLWAY,
    "Foyer": HALLWAY,
    "Corridor": HALLWAY,

    # Merge storage and service-space labels into the utility class.
    # utility/storage family
    "Storage": UTILITY,
    "Closet": UTILITY,
    "WalkIn": UTILITY,
    "Laundry": UTILITY,
    "Utility": UTILITY,
    "Pantry": UTILITY,

    # Preserve walls as a separate structural class.
    # structure
    "Wall": WALL,

        # ---- added from unknown tokens ----

    # Assign general-purpose room labels to the living-space class.
    # room-like spaces
    "Office": LIVING,
    "Den": LIVING,
    "Room": LIVING,

    # Treat additional entrance-related spaces as circulation.
    # entrance / circulation
    "DraughtLobby": HALLWAY,

    # Group additional service and storage-related spaces as utility areas.
    # utility-like spaces
    "Garage": UTILITY,
    "CarPort": UTILITY,
    "TechnicalRoom": UTILITY,
    "DressingRoom": UTILITY,

    # Exclude outdoor space from the semantic room classes.
    # outdoor (we treat as background / ignore)
    "Outdoor": BG,

    # Map non-room structural or opening elements to background so they
    # do not introduce additional semantic room categories.
    # structure/elements we choose to ignore (map to BG so they don't become rooms)
    "Door": BG,
    "Window": BG,
    "Railing": BG,
    "Column": BG,
    "Threshold": BG,
    "Glass": BG,

    # Ignore annotation, drawing and interface artefacts that do not
    # correspond to the semantic room categories used by the model.
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

    # Map less frequent room-like labels to the closest retained
    # semantic category in the reduced class representation.
    # extra room-like tokens (map to utility or living)
    "Alcove": UTILITY,
    "Attic": UTILITY,
    "Basement": UTILITY,
    "Elevated": HALLWAY,         # Treat elevated circulation space as hallway.
    "Elevator": HALLWAY,        # circulation element
    "Landing": HALLWAY,
    "Library": LIVING,
    "RecreationRoom": LIVING,
    "SwimmingPool": UTILITY,     # Group this infrequent non-primary space with utility.

    # Exclude stair-related geometry from the retained semantic classes.
    # stairs / geometry (ignore for now)
    "Stairs": BG,
    "Winding": BG,
}