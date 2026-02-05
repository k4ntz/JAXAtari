import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from functools import partial
from typing import NamedTuple, Tuple, Optional

import chex
import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer


#constants
WIDTH, HEIGHT = 160, 210  
ALPHABET_SIZE = 26
PAD_TOKEN = 26
L_MAX = 8

MAX_MISSES = 11
STEP_PENALTY = 0.0
DIFFICULTY_MODE = "B"
TIMER_SECONDS = 20
STEPS_PER_SECOND = 30

#colours of environment
BG_COLOR    = jnp.array([150, 40, 30],   dtype=jnp.uint8)
BLUE_COLOR  = jnp.array([110, 170, 240], dtype=jnp.uint8)
GOLD_COLOR  = jnp.array([213, 130, 74],  dtype=jnp.uint8)
WHITE_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

#the gallows layoutt
GALLOWS_X = 18
GALLOWS_Y = 58
GALLOWS_THICK = 6
GALLOWS_TOP_LEN = 88
GALLOWS_POST_H = 42
ROPE_X = GALLOWS_X + GALLOWS_TOP_LEN - GALLOWS_THICK - 4
ROPE_W = 4
ROPE_H = 30
ROPE_TOP_Y = GALLOWS_Y - GALLOWS_THICK

#the underscores layout
UND_Y   = 190
UND_W   = 12
UND_H   = 6
UND_GAP = 6

#glyphs(chatgpt term) layout/ can also be called characters
GLYPH_ROWS = 7
GLYPH_COLS = 5
GLYPH_SCALE_SMALL   = 2   
GLYPH_SCALE_PREVIEW = 3   

# A..Z. '1' = filled pixel, '0' = empty
# each entry is 7 strings of length 5
# not sure if this is the best way to store the glyphs, but it works, need feedback
_FONT_5x7 = {
"A": ["01110","10001","10001","11111","10001","10001","10001"],
"B": ["11110","10001","11110","10001","10001","11110","00000"],
"C": ["01111","10000","10000","10000","10000","01111","00000"],
"D": ["11110","10001","10001","10001","10001","11110","00000"],
"E": ["11111","10000","11110","10000","10000","11111","00000"],
"F": ["11111","10000","11110","10000","10000","10000","00000"],
"G": ["01111","10000","10000","10111","10001","01111","00000"],
"H": ["10001","10001","11111","10001","10001","10001","00000"],
"I": ["01110","00100","00100","00100","00100","01110","00000"],
"J": ["00001","00001","00001","00001","10001","01110","00000"],
"K": ["10001","10010","11100","10010","10001","10001","00000"],
"L": ["10000","10000","10000","10000","10000","11111","00000"],
"M": ["10001","11011","10101","10001","10001","10001","00000"],
"N": ["10001","11001","10101","10011","10001","10001","00000"],
"O": ["01110","10001","10001","10001","10001","01110","00000"],
"P": ["11110","10001","10001","11110","10000","10000","00000"],
"Q": ["01110","10001","10001","10001","10101","01110","00001"],
"R": ["11110","10001","10001","11110","10010","10001","00000"],
"S": ["01111","10000","01110","00001","00001","11110","00000"],
"T": ["11111","00100","00100","00100","00100","00100","00000"],
"U": ["10001","10001","10001","10001","10001","01110","00000"],
"V": ["10001","10001","10001","10001","01010","00100","00000"],
"W": ["10001","10001","10001","10101","11011","10001","00000"],
"X": ["10001","01010","00100","00100","01010","10001","00000"],
"Y": ["10001","01010","00100","00100","00100","00100","00000"],
"Z": ["11111","00010","00100","01000","10000","11111","00000"],
}

# again, not sure if this is the best way to store the digits, but it works, feedback needed
_DIGITS_5x7 = {
    "0": ["01110","10001","10011","10101","11001","10001","01110"],
    "1": ["00100","01100","00100","00100","00100","00100","01110"],
    "2": ["01110","10001","00001","00010","00100","01000","11111"],
    "3": ["11110","00001","00001","01110","00001","00001","11110"],
    "4": ["00010","00110","01010","10010","11111","00010","00010"],
    "5": ["11111","10000","11110","00001","00001","10001","01110"],
    "6": ["00110","01000","10000","11110","10001","10001","01110"],
    "7": ["11111","00001","00010","00100","01000","01000","01000"],
    "8": ["01110","10001","10001","01110","10001","10001","01110"],
    "9": ["01110","10001","10001","01111","00001","00010","01100"],
}

DIGIT_GLYPHS = jnp.array(
    [[[int(c) for c in row] for row in _DIGITS_5x7[ch]]
     for ch in "0123456789"],
    dtype=jnp.uint8
)

#build a (26, 7, 5) uint8 array in A to Z order
GLYPHS = jnp.array(
    [[[int(c) for c in row] for row in _FONT_5x7[chr(ord('A') + i)]]
     for i in range(26)],
    dtype=jnp.uint8
)


GOLD_SQ = 10
GOLD_XL = 2
GOLD_XR = WIDTH - GOLD_SQ - 2
GOLD_Y  = 2

#body parts
HEAD_SIZE = 12
HEAD_X = ROPE_X + (ROPE_W // 2) - (HEAD_SIZE // 2)
HEAD_Y = ROPE_TOP_Y + ROPE_H

TORSO_W = 6
TORSO_H = 26
TORSO_X = HEAD_X + (HEAD_SIZE // 2) - (TORSO_W // 2)
TORSO_Y = HEAD_Y + HEAD_SIZE

ARM_H = 6
ARM_W = 18
ARM_Y = TORSO_Y + 4
ARM_L_X = TORSO_X - ARM_W
ARM_R_X = TORSO_X + TORSO_W

LEG_W = 6
LEG_H = 18
LEG_Y = TORSO_Y + TORSO_H
LEG_L_X = TORSO_X - 4
LEG_R_X = TORSO_X + TORSO_W - 2

#lives bar 
PIP_N   = MAX_MISSES        
PIP_W   = 6
PIP_H   = 8
PIP_GAP = 4
PIP_TOTAL = PIP_N * PIP_H + (PIP_N - 1) * PIP_GAP     

PIP_X   = WIDTH - PIP_W - 4                           
PIP_Y0  = HEIGHT - PIP_TOTAL - 10                 

RIGHT_MARGIN = 10
PREVIEW_W = GLYPH_COLS * GLYPH_SCALE_PREVIEW + 8
PREVIEW_H = GLYPH_ROWS * GLYPH_SCALE_PREVIEW + 8
PREVIEW_X = PIP_X - PREVIEW_W - 8
PREVIEW_Y = 120
DRAW_PREVIEW_BORDER = False
    

#scoreboard positions
SCORE_X = GOLD_XL + GOLD_SQ + 4   
SCORE_Y = GOLD_Y
ROUND_RIGHT_X = GOLD_XR - 2       
ROUND_Y       = GOLD_Y 
SCORE_SCALE = 2

# timer
TIMER_W = 40
TIMER_H = 4
TIMER_X = WIDTH//2 - TIMER_W//2
TIMER_Y = GOLD_Y + GOLD_SQ + 4


# list of words used in the game
# not sure if this is the best way to store the words, feedback needed
# _WORDS = ["CAT", "TREE", "MOUSE", "ROBOT", "LASER", "JAX"]
_WORDS: tuple[str, ...] = (
    "ABLE", "ABOUT", "ABOVE", "ABSORB", "ABUSE", "ACADEMY", "ACTION", "ACTOR", "ACUTE", "ADAPT", "ADDER", "ADDON",
    "ADEPT", "ADJUST", "ADMIRE", "ADOBE", "ADOPT", "ADORE", "ADULT", "AFFAIR", "AFFORD", "AFRAID", "AFRESH",
    "AGENT", "AGILE", "AGING", "AGREE", "AHEAD", "AIDED", "AIMED", "AIR", "AIRPORT", "AISLE", "ALARM", "ALBUM",
    "ALERT", "ALGAE", "ALIBI", "ALLEY", "ALLOY", "ALMOND", "ALMOST", "ALONE", "ALONG", "ALPHA", "ALTAR", "ALTER",
    "ALWAYS", "AMBER", "AMBLE", "AMEND", "AMONG", "AMOUNT", "AMPLE", "ANCHOR", "ANCIENT", "ANGLE", "ANGRY",
    "ANIMAL", "ANKLE", "ANNEX", "ANNUAL", "ANSWER", "ANT", "ANTELOPE", "ANTLER", "ANVIL", "AORTA", "APART",
    "APE", "APEX", "APPLE", "APPLY", "APRIL", "APRON", "ARCADE", "ARCH", "ARCTIC", "ARENA", "ARGUE", "ARISE",
    "ARM", "ARMOR", "ARMY", "AROMA", "ARROW", "ART", "ARTERY", "ARTIST", "ASCEND", "ASH", "ASIDE", "ASK",
    "ASLEEP", "ASPECT", "ASPEN", "ASSIST", "ASSORT", "ASSURE", "ASTRAL", "ATLAS", "ATOM", "ATOMIC", "ATONCE",
    "ATTACH", "ATTIC", "AUDIO", "AUGUST", "AUNT", "AURORA", "AUTHORS", "AUTO", "AUTUMN", "AUX", "AVENUE", "AVID",
    "AVOID", "AWAKE", "AWARE", "AWARD", "AWOKE", "AXIS", "AZURE", "BABY", "BACK", "BACON", "BADGER", "BAG",
    "BAKER", "BALANCE", "BALD", "BALLET", "BALLOON", "BALMY", "BAMBOO", "BANANA", "BAND", "BANDIT", "BANK",
    "BAR", "BARB", "BARBER", "BARE", "BARGE", "BARN", "BARRACK", "BARREL", "BASALT", "BASE", "BASIC", "BASIL",
    "BASKET", "BASS", "BATCH", "BATON", "BAT", "BATTLE", "BAY", "BEACH", "BEACON", "BEAD", "BEAK", "BEAM",
    "BEAN", "BEAR", "BEAVER", "BECOME", "BED", "BEE", "BEEF", "BEET", "BEETLE", "BEGAN", "BEGIN", "BEGUN",
    "BEING", "BELIEF", "BELL", "BELLY", "BELT", "BENCH", "BERRY", "BERTH", "BEST", "BET", "BETA", "BETTER",
    "BEYOND", "BIBLE", "BICYCLE", "BID", "BIG", "BIGGER", "BIKE", "BILL", "BIN", "BIND", "BIRD", "BIRTH",
    "BISON", "BISTRO", "BIT", "BITE", "BITTER", "BLACK", "BLADE", "BLAME", "BLANK", "BLAST", "BLAZE", "BLEAK",
    "BLEED", "BLEND", "BLESS", "BLIMP", "BLIND", "BLINK", "BLIP", "BLISS", "BLOOM", "BLOSSOM", "BLOW", "BLUE",
    "BLUFF", "BLUNT", "BLUR", "BOARD", "BOAT", "BODY", "BOIL", "BOLD", "BOLT", "BOMB", "BOND", "BONE", "BONUS",
    "BOOK", "BOOST", "BOOT", "BOOTH", "BOOTS", "BORDER", "BORE", "BORN", "BORROW", "BOSS", "BOTANY", "BOTH",
    "BOTTLE", "BOTTOM", "BOUNCE", "BOUND", "BOUT", "BOWL", "BOX", "BOXER", "BOY", "BRACE", "BRANCH", "BRAND",
    "BRASS", "BRAVE", "BRAVO", "BREAD", "BREAK", "BREAST", "BREATH", "BREEZE", "BREW", "BRICK", "BRIDE",
    "BRIDGE", "BRIEF", "BRIGHT", "BRIM", "BRING", "BRISK", "BRISTLE", "BROAD", "BROIL", "BROKE", "BRONZE",
    "BROOK", "BROOM", "BROWN", "BROWSE", "BRUSH", "BRUTE", "BUBBLE", "BUCK", "BUD", "BUDDY", "BUDGET", "BUFF",
    "BUFFET", "BUG", "BUGGY", "BUILD", "BUILT", "BULB", "BULGE", "BULK", "BULL", "BULLET", "BUMPER", "BUNCH",
    "BUNDLE", "BUNNY", "BUNT", "BURDEN", "BURGER", "BURROW", "BURST", "BUS", "BUSH", "BUSY", "BUTCHER", "BUTTER",
    "BUTTON", "BUY", "BUZZ", "CAB", "CABIN", "CABLE", "CACHE", "CACTUS", "CAGE", "CAKE", "CALF", "CALL",
    "CALM", "CAMEL", "CAMP", "CANAL", "CANARY", "CANCEL", "CANDLE", "CANDY", "CANOE", "CANON", "CANOPY",
    "CANTEEN", "CANVAS", "CANYON", "CAP", "CAPABLE", "CAPE", "CAPTAIN", "CAR", "CARBON", "CARD", "CARE",
    "CARGO", "CARPET", "CARRIAGE", "CARRY", "CARROT", "CART", "CARVE", "CASE", "CASH", "CASINO", "CASK",
    "CAST", "CASTLE", "CASUAL", "CAT", "CATCH", "CATER", "CATTLE", "CAUSE", "CAVE", "CEASE", "CEDAR", "CEILING",
    "CELERY", "CELL", "CEMENT", "CENSUS", "CENTER", "CENTRE", "CHAIN", "CHAIR", "CHALK", "CHAMP", "CHANCE",
    "CHANGE", "CHAOS", "CHAPEL", "CHARM", "CHART", "CHASE", "CHASM", "CHEAP", "CHEAT", "CHEESE", "CHEF",
    "CHEMIST", "CHERRY", "CHEST", "CHEW", "CHICK", "CHIEF", "CHILD", "CHILI", "CHILL", "CHIME", "CHIN",
    "CHINA", "CHIP", "CHOCOL", "CHOICE", "CHOKE", "CHORD", "CHORE", "CHOSE", "CHOSEN", "CHUCK", "CHUNK",
    "CHURCH", "CIDER", "CINEMA", "CIPHER", "CIRCLE", "CIRCUIT", "CIRCUS", "CITY", "CIVIC", "CLAIM", "CLAM",
    "CLAMP", "CLAN", "CLASH", "CLASP", "CLASS", "CLAUSE", "CLAW", "CLAY", "CLEAN", "CLEAR", "CLERK", "CLICK",
    "CLIFF", "CLIMB", "CLINIC", "CLIP", "CLOCK", "CLOSE", "CLOSET", "CLOUD", "CLOUT", "CLOVE", "CLOWN", "CLUB",
    "CLUE", "CLUMP", "COACH", "COAL", "COAST", "COAT", "COAX", "COBALT", "COBRA", "COCOA", "COCONUT", "CODE",
    "CODER", "COFFEE", "COIL", "COIN", "CONE", "CONFIRM", "CONGA", "CONIC", "CONN", "CONTOUR", "CONTRA",
    "CONTROL", "CONVEY", "COOK", "COOKIE", "COOL", "COOP", "COPPER", "COPY", "CORAL", "CORD", "CORE",
    "CORK", "CORN", "CORNER", "CORRAL", "COST", "COTTON", "COUCH", "COUGAR", "COUNT", "COUNTRY", "COUPLE",
    "COURSE", "COURT", "COVER", "COW", "COWBOY", "COYOTE", "CRAB", "CRACK", "CRAFT", "CRANE", "CRASH",
    "CRATE", "CRAWL", "CRAYON", "CRAZE", "CRAZY", "CREAM", "CREATE", "CREDIT", "CREEK", "CREEP", "CREST",
    "CREW", "CRIB", "CRICKET", "CRIME", "CRISP", "CROOK", "CROP", "CROSS", "CROWD", "CROWN", "CRUDE",
    "CRUEL", "CRUISE", "CRUMB", "CRUSH", "CRUST", "CRY", "CUBE", "CUBIC", "CUBS", "CUFF", "CUISINE",
    "CULT", "CULTURE", "CUP", "CURB", "CURE", "CURL", "CURRENCY", "CURRY", "CURSE", "CURVE", "CUSHION",
    "CUSTOM", "CUT", "CYCLE", "CYCLER", "CYCLIC", "CYLINDER", "DAD", "DAILY", "DAIRY", "DAISY", "DAMAGE",
    "DANCE", "DANCER", "DANGER", "DARING", "DARK", "DARLING", "DART", "DATA", "DATE", "DAWN", "DAY",
    "DEAL", "DEAR", "DEBRIS", "DEBT", "DEBUG", "DEBUT", "DECENT", "DECIDE", "DECK", "DECOR", "DEED",
    "DEEP", "DEER", "DEFEND", "DEFER", "DEFINE", "DEGREE", "DELAY", "DELTA", "DELVE", "DEMAND", "DEMN",
    "DEMO", "DENT", "DENTAL", "DENTIST", "DENY", "DEPART", "DEPEND", "DEPLOY", "DEPOSIT", "DEPTH",
    "DEPUTY", "DERBY", "DESERT", "DESIGN", "DESK", "DESPITE", "DETAIL", "DETOUR", "DEUCE", "DEVICE", "DEVIL",
    "DEVOTE", "DIARY", "DICE", "DIE", "DIET", "DIG", "DIGEST", "DIGIT", "DILUTE", "DIM", "DINER", "DINGO",
    "DINGY", "DIODE", "DIP", "DIPPER", "DIRECT", "DIRT", "DIRTY", "DISCO", "DISH", "DISK", "DIVE",
    "DIVER", "DIVIDE", "DIVING", "DIZZY", "DOCK", "DOCTOR", "DODGE", "DOG", "DOGMA", "DOLL", "DOLPHIN",
    "DOMAIN", "DOME", "DOMINO", "DONE", "DONKEY", "DONOR", "DOOR", "DOSE", "DOT", "DOUBLE", "DOUGH",
    "DOVE", "DOWN", "DOZEN", "DRAB", "DRAFT", "DRAGON", "DRAIN", "DRAMA", "DRANK", "DRAPE", "DRAW",
    "DRAWN", "DREAD", "DREAM", "DRESS", "DRIED", "DRIFT", "DRILL", "DRINK", "DRIVE", "DRIVER", "DROID",
    "DROOP", "DROP", "DROVE", "DROWN", "DRUG", "DRUM", "DRY", "DUCK", "DUCT", "DUKE", "DULL", "DUMB",
    "DUNE", "DUNK", "DUO", "DUSK", "DUST", "DUTY", "DWELL", "DWINDLE", "EACH", "EAGER", "EAGLE", "EAR",
    "EARLY", "EARN", "EARTH", "EASE", "EASILY", "EAST", "EASY", "EAT", "EBONY", "ECHO", "ECLIPSE", "ECO",
    "EDGE", "EDIT", "EDITOR", "EEL", "EFFECT", "EFFORT", "EGG", "EIGHT", "EITHER", "ELBOW", "ELDER",
    "ELECT", "ELEGANT", "ELEMENT", "ELEPHANT", "ELEVATE", "ELITE", "ELK", "ELM", "ELSE", "EMBED", "EMBER",
    "EMBRACE", "EMIT", "EMOTION", "EMPLOY", "EMPTY", "EMU", "ENABLE", "ENACT", "END", "ENDURE", "ENEMY",
    "ENERGY", "ENJOY", "ENLIST", "ENORMOUS", "ENOUGH", "ENROLL", "ENSURE", "ENTER", "ENTIRE", "ENTRY",
    "ENVOY", "ENZYME", "EPOCH", "EQUAL", "EQUIP", "ERA", "ERASE", "ERECT", "ERODE", "ERROR", "ESCAPE",
    "ESSAY", "ESTATE", "ETC", "ETHIC", "ETHOS", "EURO", "EVADE", "EVEN", "EVENT", "EVER", "EVICT", "EVIL",
    "EVOLVE", "EXACT", "EXAM", "EXCEL", "EXHALE", "EXIST", "EXIT", "EXOTIC", "EXPAND", "EXPECT", "EXPENSE",
    "EXPERT", "EXPIRE", "EXPORT", "EXPOSE", "EXTEND", "EXTRA", "EYE", "FABRIC", "FACE", "FACT", "FACTOR",
    "FACTORY", "FADE", "FAIL", "FAIR", "FAITH", "FAKE", "FALL", "FALSE", "FAME", "FAMOUS", "FAN", "FANCY",
    "FANG", "FARM", "FARMER", "FARMS", "FAST", "FATAL", "FATE", "FAULT", "FAUNA", "FAVOR", "FAX", "FEAST",
    "FEED", "FEEL", "FELT", "FENCE", "FERN", "FERRY", "FERTILE", "FEST", "FETCH", "FEVER", "FEW", "FIBER",
    "FICTION", "FIELD", "FIERCE", "FIFTH", "FIFTY", "FIG", "FIGHT", "FIGURE", "FILE", "FILL", "FILM", "FILTER",
    "FINAL", "FINCH", "FIND", "FINE", "FINISH", "FIRE", "FIRM", "FIRST", "FISH", "FISHER", "FIST", "FIT",
    "FIVE", "FIX", "FIXED", "FLAG", "FLAKY", "FLAME", "FLANK", "FLASH", "FLAT", "FLAW", "FLEA", "FLEET",
    "FLESH", "FLICK", "FLIER", "FLIGHT", "FLING", "FLINT", "FLIP", "FLOAT", "FLOCK", "FLOOD", "FLOOR", "FLOUR",
    "FLOW", "FLOWER", "FLU", "FLUFF", "FLUID", "FLUTE", "FLUX", "FLY", "FOAM", "FOCUS", "FOG", "FOGGY",
    "FOOD", "FOOL", "FOOT", "FORCE", "FORD", "FOREST", "FORGET", "FORK", "FORM", "FORMAT", "FORT", "FORUM",
    "FOSSIL", "FOUND", "FOUR", "FOX", "FRAME", "FRANK", "FRAUD", "FRESH", "FRIAR", "FRIDAY", "FRIED",
    "FRIEND", "FRIES", "FROG", "FROM", "FRONT", "FROST", "FROZE", "FRUIT", "FUEL", "FULL", "FUN", "FUNCTION",
    "FUND", "FUNNY", "FUR", "FUSION", "FUTURE", "GADGET", "GALE", "GALLON", "GAMBLE", "GAME", "GAMER",
    "GAMMA", "GANG", "GARDEN", "GARLIC", "GASH", "GATE", "GATHER", "GAUGE", "GAUNT", "GAZE", "GEAR", "GECKO",
    "GEL", "GEM", "GENE", "GENERIC", "GENIUS", "GENRE", "GENTLE", "GENTLY", "GENUS", "GERM", "GET", "GHOST",
    "GIANT", "GIFT", "GIG", "GIGGLE", "GIRAFFE", "GIRL", "GIST", "GIVE", "GLACIER", "GLAD", "GLANCE", "GLASS",
    "GLAZE", "GLEAM", "GLIDE", "GLINT", "GLOBE", "GLOOM", "GLORY", "GLOVE", "GLOW", "GLUE", "GOAL", "GOAT",
    "GOBLIN", "GOD", "GODS", "GOING", "GOLD", "GOLF", "GONDOLA", "GONE", "GOOD", "GOOSE", "GOPHER", "GORILLA",
    "GOSPEL", "GOT", "GOURD", "GOWN", "GRACE", "GRADE", "GRAIN", "GRAND", "GRANT", "GRAPE", "GRAPH", "GRASP",
    "GRASS", "GRATE", "GRAVEL", "GRAVY", "GREAT", "GREED", "GREEN", "GREET", "GREY", "GRID", "GRIEF", "GRILL",
    "GRIN", "GRIND", "GRIP", "GRIT", "GROOM", "GROSS", "GROUP", "GROVE", "GROW", "GROWN", "GRUB", "GRUNT",
    "GUARD", "GUAVA", "GUESS", "GUEST", "GUIDE", "GUILD", "GUILT", "GUITAR", "GULL", "GULP", "GUM", "GUMBO",
    "GUST", "GUT", "GUY", "GYM", "GYPSY", "HABIT", "HACK", "HAIL", "HAIR", "HALF", "HALL", "HALO", "HALT",
    "HAM", "HAMLET", "HAND", "HANDLE", "HANG", "HARBOR", "HARD", "HARE", "HARM", "HARP", "HARSH", "HARVEST",
    "HAS", "HASH", "HASTE", "HAT", "HATCH", "HATE", "HAVE", "HAWK", "HAY", "HAZEL", "HAZY", "HEAD", "HEAL",
    "HEAP", "HEAR", "HEARD", "HEART", "HEAT", "HEAVY", "HEDGE", "HEEL", "HEFT", "HEIGHT", "HEIR", "HELD",
    "HELIX", "HELLO", "HELM", "HELMET", "HELP", "HEMP", "HEN", "HERB", "HERD", "HERON", "HERO", "HESSIAN",
    "HIDDEN", "HIDE", "HIGH", "HIKER", "HILL", "HILT", "HIND", "HINGE", "HINT", "HIP", "HIRE", "HIS", "HIT",
    "HIVE", "HOBBY", "HOG", "HOLD", "HOLE", "HOLIDAY", "HOLLY", "HOME", "HONEST", "HONEY", "HONOR", "HOOD",
    "HOOK", "HOOP", "HOP", "HOPPER", "HORN", "HORNET", "HORSE", "HOSE", "HOST", "HOT", "HOTEL", "HOUR", "HOUSE",
    "HOVER", "HUGE", "HUMAN", "HUMBLE", "HUMID", "HUMOR", "HUNGER", "HUNT", "HURRY", "HURT", "HUSK", "HUSKY",
    "HUT", "HYBRID", "HYDRA", "HYENA", "HYMNS", "ICE", "ICING", "ICON", "IDEA", "IDEAL", "IDENTITY", "IDLE",
    "IDOL", "IGLOO", "IGNITE", "IGUANA", "ILL", "IMAGE", "IMPACT", "IMPORT", "IN", "INCH", "INDEX", "INDIGO",
    "INDOOR", "INEPT", "INFER", "INFO", "INGOT", "INJECT", "INK", "INLAY", "INLET", "INN", "INNER", "INPUT",
    "INSIDE", "INSIGHT", "INSPIRE", "INSTANCE", "INTAKE", "INTENT", "INTERN", "INTO", "INVENT", "INVEST",
    "ION", "IRON", "ISLAND", "IVORY", "IVY", "JACK", "JACKET", "JAIL", "JAM", "JAR", "JASMINE", "JAW", "JAZZ",
    "JEANS", "JEEP", "JELLY", "JERKY", "JET", "JEWEL", "JIG", "JINX", "JOB", "JOG", "JOIN", "JOINT", "JOKE",
    "JOLLY", "JOURNAL", "JOY", "JUICE", "JUICY", "JULY", "JUMP", "JUMPER", "JUNE", "JUNGLE", "JUNIOR", "JUNK",
    "JUROR", "JUST", "KAYAK", "KEEN", "KEEP", "KELP", "KERNEL", "KETTLE", "KEY", "KEYBOARD", "KICK", "KID",
    "KIDNEY", "KILO", "KIND", "KING", "KIOSK", "KISS", "KIT", "KITE", "KITTEN", "KIWI", "KNEAD", "KNEE",
    "KNIFE", "KNIT", "KNOB", "KNOT", "KNOW", "KOALA", "LABEL", "LABOR", "LACE", "LACK", "LADDER", "LADLE",
    "LADY", "LAGOON", "LAKE", "LAMB", "LAMP", "LANCE", "LAND", "LANE", "LAP", "LAPTOP", "LARGE", "LARK",
    "LASER", "LAST", "LATCH", "LATE", "LAUGH", "LAUNCH", "LAVA", "LAWN", "LAWSUIT", "LAWYER", "LAYER", "LAZY",
    "LEAF", "LEAGUE", "LEAK", "LEAN", "LEARN", "LEASE", "LEASH", "LEAVE", "LED", "LEFT", "LEG", "LEGAL",
    "LEGEND", "LEMON", "LEND", "LENS", "LEOPARD", "LESS", "LETTUCE", "LEVEL", "LEVER", "LIAR", "LID", "LIE",
    "LIFE", "LIFT", "LIGHT", "LIKE", "LILAC", "LILY", "LIMB", "LIME", "LIMIT", "LINE", "LINK", "LION",
    "LIP", "LIQUID", "LIST", "LIT", "LITER", "LITTLE", "LIVE", "LIVER", "LIZARD", "LOAD", "LOAF", "LOAN",
    "LOBSTER", "LOCAL", "LOCK", "LODGE", "LOFT", "LOG", "LOOM", "LOON", "LOOP", "LOOSE", "LOOT", "LORD",
    "LORRY", "LOSE", "LOSS", "LOST", "LOTION", "LOTUS", "LOUD", "LOUNGE", "LOVE", "LOW", "LOYAL", "LUCK",
    "LUCID", "LUMP", "LUNCH", "LUNGE", "LUSH", "LUST", "LUTE", "LUXURY", "LYCHEE", "LYRIC", "MAGENTA", "MAGIC",
    "MAID", "MAIL", "MAJOR", "MAKER", "MALE", "MALL", "MAMMAL", "MAN", "MANGO", "MANTIS", "MAP", "MAPLE",
    "MARBLE", "MARCH", "MARE", "MARGIN", "MARINE", "MARK", "MARKET", "MARROW", "MARRY", "MARS", "MART",
    "MASK", "MASS", "MASTER", "MATCH", "MATE", "MATH", "MATRIX", "MAY", "MAYOR", "MAZE", "MEADOW", "MEAL",
    "MEAN", "MEASURE", "MEAT", "MECHANIC", "MEDAL", "MEDIA", "MEDIC", "MEET", "MELON", "MELT", "MEMBER",
    "MEMORY", "MEND", "MENU", "MERCY", "MERGE", "MERIT", "MERRY", "MESS", "METAL", "METER", "METRO", "MICRO",
    "MIDDLE", "MIGHT", "MILD", "MILE", "MILK", "MILL", "MIMIC", "MINCE", "MIND", "MINER", "MINI", "MINOR",
    "MINT", "MINUTE", "MIRROR", "MIRTH", "MIX", "MIXER", "MIXED", "MIXTURE", "MOBILE", "MODEL", "MODEM",
    "MODERN", "MODIFY", "MODULE", "MOIST", "MOLD", "MOLAR", "MOM", "MONITOR", "MONKEY", "MONTH", "MOOD",
    "MOON", "MOOR", "MOOSE", "MOP", "MORAL", "MORE", "MORNING", "MORPH", "MOSS", "MOST", "MOTH", "MOTHER",
    "MOTION", "MOTOR", "MOTTO", "MOUND", "MOUNT", "MOURN", "MOUSE", "MOUTH", "MOVE", "MOVIE", "MUFFIN",
    "MULE", "MUSE", "MUSIC", "MUSK", "MUSSEL", "MUST", "MUTE", "MUTTER", "MUTTON", "NACHO", "NAIL", "NAME",
    "NAPKIN", "NARROW", "NASTY", "NATION", "NATIVE", "NATURE", "NAVY", "NEAR", "NEAT", "NECK", "NEED", "NEON",
    "NERVE", "NEST", "NET", "NETWORK", "NEURAL", "NEATLY", "NEVER", "NEW", "NEWS", "NEXT", "NICE", "NICKEL",
    "NIECE", "NIGHT", "NINE", "NINJA", "NINTH", "NOBLE", "NOD", "NOISE", "NOISY", "NONE", "NOODLE", "NORTH",
    "NOSE", "NOTCH", "NOTE", "NOTIFY", "NOTION", "NOVEL", "NOVICE", "NOW", "NUANCE", "NUCLEAR", "NUDGE",
    "NULL", "NUMBER", "NUMB", "NURSE", "NUT", "NYLON", "OAK", "OAR", "OASIS", "OAT", "OATH", "OBEY", "OBJECT",
    "OBLIGE", "OCEAN", "OCTAVE", "OCTET", "ODD", "ODOR", "OFF", "OFFER", "OFFICE", "OFFSET", "OFTEN", "OIL",
    "OINK", "OLD", "OLIVE", "OMEGA", "OMELET", "OMEN", "OMICRON", "ONCE", "ONE", "ONION", "ONLINE", "ONLY",
    "ONTO", "ONYX", "OPEN", "OPERA", "OPINE", "OPTIC", "OPTION", "ORANGE", "ORBIT", "ORCHID", "ORDER",
    "ORE", "ORGAN", "ORIGIN", "ORION", "ORNATE", "ORPHAN", "OSPREY", "OSTRICH", "OTHER", "OTTER", "OUNCE",
    "OUR", "OUT", "OVAL", "OVEN", "OVER", "OWL", "OX", "OYSTER", "OZONE", "PACE", "PACK", "PAD", "PADDLE",
    "PAGE", "PAID", "PAIL", "PAIN", "PAINT", "PAIR", "PALACE", "PALE", "PALM", "PANDA", "PANEL", "PANTRY",
    "PANTS", "PAPA", "PAPER", "PARADE", "PARCEL", "PARDON", "PARENT", "PARK", "PARROT", "PART", "PARTY",
    "PASTA", "PASTE", "PASTOR", "PASTRY", "PATCH", "PATH", "PATIENT", "PATROL", "PATIO", "PATSY", "PAUSE",
    "PAY", "PEACE", "PEACH", "PEAR", "PEARL", "PEAS", "PECK", "PEDAL", "PEEL", "PEEP", "PEER", "PELICAN",
    "PEN", "PENALTY", "PENCIL", "PEND", "PENGUIN", "PENNY", "PEOPLE", "PEPPER", "PERCH", "PERFECT", "PERIL",
    "PERK", "PERMIT", "PERSON", "PEST", "PET", "PHASE", "PHONE", "PHOTO", "PHYSIC", "PICK", "PICKUP",
    "PICNIC", "PIE", "PIECE", "PIER", "PIGEON", "PIGMENT", "PILOT", "PIN", "PINCH", "PINE", "PING", "PINK",
    "PINT", "PIONEER", "PIPE", "PIPELINE", "PIRATE", "PISTOL", "PITCH", "PIVOT", "PIXEL", "PIZZA", "PLACE",
    "PLAID", "PLAIN", "PLAN", "PLANE", "PLANET", "PLANT", "PLASMA", "PLATE", "PLAY", "PLAYER", "PLEAD",
    "PLEASE", "PLEDGE", "PLENTY", "PLIERS", "PLIGHT", "PLOD", "PLOW", "PLUCK", "PLUG", "PLUMB", "PLUME",
    "PLUMP", "PLUM",
)


def _encode_word(w: str) -> jnp.ndarray:
    arr = [ord(c) - 65 for c in w.upper()]
    arr = arr[:L_MAX]
    arr += [PAD_TOKEN] * (L_MAX - len(arr))
    return jnp.array(arr, dtype=jnp.int32)

WORDS_ENC = jnp.stack([_encode_word(w) for w in _WORDS], axis=0)
WORDS_LEN = jnp.array([min(len(w), L_MAX) for w in _WORDS], dtype=jnp.int32)
N_WORDS = WORDS_ENC.shape[0]


class HangmanConstants(NamedTuple):
    MAX_MISSES: int = MAX_MISSES
    STEP_PENALTY: float = STEP_PENALTY
    DIFFICULTY_MODE: str = DIFFICULTY_MODE
    TIMER_SECONDS: int = TIMER_SECONDS
    STEPS_PER_SECOND: int = STEPS_PER_SECOND


class HangmanState(NamedTuple):
    key: chex.Array
    word: chex.Array          
    length: chex.Array        
    mask: chex.Array          
    guessed: chex.Array       
    misses: chex.Array        
    lives: chex.Array         
    cursor_idx: chex.Array    
    done: chex.Array          
    reward: chex.Array        
    step_counter: chex.Array
    score: chex.Array   
    round_no: chex.Array
    time_left_steps: chex.Array
    cpu_score: chex.Array
    timer_max_steps: chex.Array      
    last_commit: chex.Array
  

class HangmanObservation(NamedTuple):
    revealed: chex.Array      
    mask: chex.Array          
    guessed: chex.Array       
    misses: chex.Array        
    lives: chex.Array         
    cursor_idx: chex.Array    

class HangmanInfo(NamedTuple):
    time: chex.Array

# helpers funcutions
@jax.jit
def _sample_word(key: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    key, sub = jrandom.split(key)
    idx = jrandom.randint(sub, shape=(), minval=0, maxval=N_WORDS, dtype=jnp.int32)
    return key, WORDS_ENC[idx], WORDS_LEN[idx]

@jax.jit
def _compute_revealed(word: chex.Array, mask: chex.Array) -> chex.Array:
    return jnp.where(mask.astype(bool), word, PAD_TOKEN)

def _action_delta_cursor(action: chex.Array) -> chex.Array:
    # Actions that move cursor up: UP, UPRIGHT, UPLEFT, UPFIRE, UPRIGHTFIRE, UPLEFTFIRE
    up_like = jnp.logical_or(
        jnp.logical_or(
            jnp.logical_or(action == Action.UP, action == Action.UPRIGHT),
            jnp.logical_or(action == Action.UPLEFT, action == Action.UPFIRE)
        ),
        jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE)
    )
    # Actions that move cursor down: DOWN, DOWNRIGHT, DOWNLEFT, DOWNFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE
    down_like = jnp.logical_or(
        jnp.logical_or(
            jnp.logical_or(action == Action.DOWN, action == Action.DOWNRIGHT),
            jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNFIRE)
        ),
        jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE)
    )
    return jnp.where(up_like, -1, jnp.where(down_like, 1, 0)).astype(jnp.int32)

def _action_commit(action: chex.Array) -> chex.Array:
    # All FIRE actions commit: FIRE, UPFIRE, DOWNFIRE, RIGHTFIRE, LEFTFIRE, 
    # UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE
    return jnp.logical_or(
        jnp.logical_or(
            jnp.logical_or(action == Action.FIRE, action == Action.UPFIRE),
            jnp.logical_or(action == Action.DOWNFIRE, action == Action.RIGHTFIRE)
        ),
        jnp.logical_or(
            jnp.logical_or(action == Action.LEFTFIRE, action == Action.UPRIGHTFIRE),
            jnp.logical_or(action == Action.UPLEFTFIRE, 
                jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE))
        )
    )

@jax.jit
def _advance_cursor_skip_guessed(cursor: chex.Array,
                                 delta: chex.Array,
                                 guessed: chex.Array) -> chex.Array:
    """Move one step in the delta direction, then keep stepping while guessed[cur]==1.
       Bounded by 26 steps to avoid infinite loops if all letters are guessed."""
    # First step (if delta==0, stay)
    step = jnp.where(delta > 0, 1, jnp.where(delta < 0, -1, 0)).astype(jnp.int32)
    cur0 = jnp.where(step == 0, cursor, (cursor + step) % ALPHABET_SIZE)

    def cond_fun(carry):
        cur, n = carry
        need_move = jnp.logical_and(step != 0, guessed[cur] == 1)
        return jnp.logical_and(n < ALPHABET_SIZE, need_move)

    def body_fun(carry):
        cur, n = carry
        return ((cur + step) % ALPHABET_SIZE, n + 1)

    cur, _ = lax.while_loop(cond_fun, body_fun, (cur0, jnp.int32(0)))
    return cur
    

def _draw_rect(r, x, y, w, h, color):
    tile = jnp.broadcast_to(color, (int(h), int(w), 3))
    return lax.dynamic_update_slice(r, tile, (jnp.int32(y), jnp.int32(x), jnp.int32(0)))

def _draw_outline(r, x, y, w, h, color, t=1):
    #top
    r = _draw_rect(r, x, y, w, t, color)                 
    #bottom
    r = _draw_rect(r, x, y + h - t, w, t, color)         
    
    #left 
    r = _draw_rect(r, x, y, t, h, color)                 
    
    #rifht
    r = _draw_rect(r, x + w - t, y, t, h, color)         
    return r

def _draw_if(cond, fn, r):
    return lax.cond(cond, lambda rr: fn(rr), lambda rr: rr, r)

def _draw_glyph_bitmap(r, x, y, bitmap, scale, color):
    """bitmap: (7,5) uint8 0/1. Scales each pixel to (scale x scale)."""
    def row_loop(i, rr):
        def col_loop(j, r2):
            on = bitmap[i, j] == 1
            return lax.cond(
                on,
                lambda r3: _draw_rect(r3, x + j*scale, y + i*scale, scale, scale, color),
                lambda r3: r3,
                r2
            )
        return lax.fori_loop(0, GLYPH_COLS, col_loop, rr)
    return lax.fori_loop(0, GLYPH_ROWS, row_loop, r)

def _draw_glyph_idx(r, x, y, idx, scale, color):
    bitmap = GLYPHS[jnp.clip(idx, 0, 25)]
    return _draw_glyph_bitmap(r, x, y, bitmap, scale, color)


def _draw_digit_idx(r, x, y, idx, scale, color):
    bitmap = DIGIT_GLYPHS[jnp.clip(idx, 0, 9)]
    return _draw_glyph_bitmap(r, x, y, bitmap, scale, color)

def _draw_number_left(r, x, y, val, scale, color, max_digits=4):
    w = GLYPH_COLS * scale
    gap = 2
    a = jnp.maximum(val, 0)

    # least to most
    d0 = a % 10
    d1 = (a // 10)   % 10
    d2 = (a // 100)  % 10
    d3 = (a // 1000) % 10

    digits = jnp.array([d3, d2, d1, d0], dtype=jnp.int32)

    n = jnp.maximum(1, jnp.where(a >= 1000, 4, jnp.where(a >= 100, 3, jnp.where(a >= 10, 2, 1))))
    start = 4 - n  

    def body(i, rr):
        xi = x + i * (w + gap)
        idx = jnp.clip(start + i, 0, 3)
        return lax.cond(i < n,
                        lambda r2: _draw_digit_idx(r2, xi, y, digits[idx], scale, color),
                        lambda r2: r2,
                        rr)
    return lax.fori_loop(0, max_digits, body, r)

def _draw_number_right(r, right_x, y, val, scale, color, max_digits=4):
    w = GLYPH_COLS * scale
    gap = 2
    a = jnp.maximum(val, 0)

    d3 = (a // 1000) % 10
    d2 = (a // 100)  % 10
    d1 = (a // 10)   % 10
    d0 = a % 10

    n = jnp.maximum(1, jnp.where(a >= 1000, 4, jnp.where(a >= 100, 3, jnp.where(a >= 10, 2, 1))))
    
    # most to least
    digits = jnp.array([d3, d2, d1, d0], dtype=jnp.int32)  
    
    #right-aligned
    mask = jnp.arange(4) >= (4 - n)

    def body(i, rr):
        i_right = 3 - i
        xi = right_x - (i + 1) * w - i * gap
        return lax.cond(mask[i_right],
                        lambda r2: _draw_digit_idx(r2, xi, y, digits[i_right], scale, color),
                        lambda r2: r2,
                        rr)
    return lax.fori_loop(0, max_digits, body, r)

# render
class HangmanRenderer(JAXGameRenderer):
    def __init__(self):
        pass

    # @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        
        raster = jnp.broadcast_to(BG_COLOR, (HEIGHT, WIDTH, 3))

        # gold accents
        raster = _draw_rect(raster, GOLD_XL, GOLD_Y, GOLD_SQ, GOLD_SQ, GOLD_COLOR)
        raster = _draw_rect(raster, GOLD_XR, GOLD_Y, GOLD_SQ, GOLD_SQ, GOLD_COLOR)

        # gallows + rope
        raster = _draw_rect(raster, GALLOWS_X, GALLOWS_Y, GALLOWS_THICK, GALLOWS_POST_H, BLUE_COLOR)
        raster = _draw_rect(raster, GALLOWS_X, GALLOWS_Y - GALLOWS_THICK, GALLOWS_TOP_LEN, GALLOWS_THICK, BLUE_COLOR)
        raster = _draw_rect(raster, ROPE_X, ROPE_TOP_Y, ROPE_W, ROPE_H, BLUE_COLOR)

        #underscores (bottom-center
        length  = state.length
        total_w = length * UND_W + jnp.maximum(length - 1, 0) * UND_GAP
        start_x = (WIDTH - total_w) // 2

        def draw_underscore(i, r):
            x = start_x + i * (UND_W + UND_GAP)
            return lax.cond(
                i < length,
                lambda rr: _draw_rect(rr, x, UND_Y, UND_W, UND_H, BLUE_COLOR),
                lambda rr: rr,
                r
            )
        raster = lax.fori_loop(0, L_MAX, draw_underscore, raster)

        #revealed letters above underscores chars
        #not working 
        glyph_w = GLYPH_COLS * GLYPH_SCALE_SMALL 
        x_inset = (UND_W - glyph_w) // 2         
        def draw_revealed(i, r):
            x = start_x + i * (UND_W + UND_GAP) + x_inset
            cond = jnp.logical_and(i < length, state.mask[i] == 1)
            idx  = state.word[i]  
            return lax.cond(
                cond,
                lambda rr: _draw_glyph_idx(rr, x, UND_Y - GLYPH_ROWS*GLYPH_SCALE_SMALL - 4, idx, GLYPH_SCALE_SMALL, BLUE_COLOR),
                lambda rr: rr,
                r
            )
        raster = lax.fori_loop(0, L_MAX, draw_revealed, raster)

        #preview letter on the right
        if DRAW_PREVIEW_BORDER:
            raster = _draw_outline(raster, PREVIEW_X, PREVIEW_Y, PREVIEW_W, PREVIEW_H, WHITE_COLOR, t=1)
        px = PREVIEW_X + (PREVIEW_W - GLYPH_COLS*GLYPH_SCALE_PREVIEW)//2
        py = PREVIEW_Y + (PREVIEW_H - GLYPH_ROWS*GLYPH_SCALE_PREVIEW)//2
        raster = _draw_glyph_idx(raster, px, py, state.cursor_idx, GLYPH_SCALE_PREVIEW, BLUE_COLOR)
        
        #lives bar
        lives_clamped = jnp.clip(state.lives, 0, PIP_N)
        def draw_pip(i, r):
            y = PIP_Y0 + i * (PIP_H + PIP_GAP)
            return lax.cond(i < lives_clamped,
                            lambda rr: _draw_rect(rr, PIP_X, y, PIP_W, PIP_H, BLUE_COLOR),
                            lambda rr: rr,
                            r)
        raster = lax.fori_loop(0, PIP_N, draw_pip, raster)

        def _draw_timer(rr):
            denom = jnp.maximum(state.timer_max_steps, 1)
            frac  = jnp.clip(state.time_left_steps.astype(jnp.float32) / denom.astype(jnp.float32), 0.0, 1.0)
            fill  = jnp.int32(jnp.round(frac * TIMER_W))

            r1 = _draw_outline(rr, TIMER_X, TIMER_Y, TIMER_W, TIMER_H, WHITE_COLOR, t=1)

            #take background slice to preserve red where dont fill
            base = lax.dynamic_slice(
                r1,
                (jnp.int32(TIMER_Y + 1), jnp.int32(TIMER_X), jnp.int32(0)),
                (TIMER_H - 2, TIMER_W, 3)
            )
            full_tile = jnp.broadcast_to(BLUE_COLOR, (TIMER_H - 2, TIMER_W, 3))
            col_mask  = (jnp.arange(TIMER_W, dtype=jnp.int32) < fill)[jnp.newaxis, :, jnp.newaxis]
            tile      = jnp.where(col_mask, full_tile, base)

            return lax.dynamic_update_slice(
                r1, tile,
                (jnp.int32(TIMER_Y + 1), jnp.int32(TIMER_X), jnp.int32(0))
            )

        # only draw bar when A-mode is active
        raster = lax.cond(state.timer_max_steps > 0, _draw_timer, lambda rr: rr, raster)


        # progressive hangman body (misses)
        # commenting out the old hangman body parts because they were not working well
        # m = jnp.clip(state.misses, 0, 6)
        # raster = _draw_if(m >= 1, lambda rr: _draw_rect(rr, HEAD_X, HEAD_Y, HEAD_SIZE, HEAD_SIZE, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 2, lambda rr: _draw_rect(rr, TORSO_X, TORSO_Y, TORSO_W, TORSO_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 3, lambda rr: _draw_rect(rr, ARM_L_X, ARM_Y, ARM_W, ARM_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 4, lambda rr: _draw_rect(rr, ARM_R_X, ARM_Y, ARM_W, ARM_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 5, lambda rr: _draw_rect(rr, LEG_L_X, LEG_Y, LEG_W, LEG_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 6, lambda rr: _draw_rect(rr, LEG_R_X, LEG_Y, LEG_W, LEG_H, BLUE_COLOR), raster)
        
        #progressive hangman body (11 parts)
        m = jnp.clip(state.misses, 0, 11)

        #head
        raster = _draw_if(m >= 1,  lambda rr: _draw_rect(rr, HEAD_X, HEAD_Y, HEAD_SIZE, HEAD_SIZE, BLUE_COLOR), raster)

        #split torso into two segments
        TORSO1_H = TORSO_H // 2
        TORSO2_H = TORSO_H - TORSO1_H

        #torso top
        raster = _draw_if(m >= 2,  lambda rr: _draw_rect(rr, TORSO_X, TORSO_Y, TORSO_W, TORSO1_H, BLUE_COLOR), raster)
        # 3) torso bottom
        raster = _draw_if(m >= 3,  lambda rr: _draw_rect(rr, TORSO_X, TORSO_Y + TORSO1_H, TORSO_W, TORSO2_H, BLUE_COLOR), raster)

        #arms split into upper + forearm
        ARM_UP_W  = 12
        ARM_LOW_W = ARM_W - ARM_UP_W

        #left upper arm
        raster = _draw_if(m >= 4,  lambda rr: _draw_rect(rr, TORSO_X - ARM_UP_W, ARM_Y, ARM_UP_W, ARM_H, BLUE_COLOR), raster)
        #left forearm
        raster = _draw_if(m >= 5,  lambda rr: _draw_rect(rr, TORSO_X - ARM_W,     ARM_Y, ARM_LOW_W, ARM_H, BLUE_COLOR), raster)

        #right upper arm
        raster = _draw_if(m >= 6,  lambda rr: _draw_rect(rr, TORSO_X + TORSO_W,            ARM_Y, ARM_UP_W,  ARM_H, BLUE_COLOR), raster)
        #right forearm
        raster = _draw_if(m >= 7,  lambda rr: _draw_rect(rr, TORSO_X + TORSO_W + ARM_UP_W, ARM_Y, ARM_LOW_W, ARM_H, BLUE_COLOR), raster)

        #legs split into thigh + shin
        LEG1_H = (LEG_H // 2) + 1
        LEG2_H = LEG_H - LEG1_H

        #left thigh
        raster = _draw_if(m >= 8,  lambda rr: _draw_rect(rr, LEG_L_X, LEG_Y, LEG_W, LEG1_H, BLUE_COLOR), raster)
        #left shin
        raster = _draw_if(m >= 9,  lambda rr: _draw_rect(rr, LEG_L_X, LEG_Y + LEG1_H, LEG_W, LEG2_H, BLUE_COLOR), raster)

        #right thigh
        raster = _draw_if(m >= 10, lambda rr: _draw_rect(rr, LEG_R_X, LEG_Y, LEG_W, LEG1_H, BLUE_COLOR), raster)
        
        raster = _draw_if(m >= 11, lambda rr: _draw_rect(rr, LEG_R_X, LEG_Y + LEG1_H, LEG_W, LEG2_H, BLUE_COLOR), raster)


        raster = _draw_number_left(
            raster, SCORE_X, SCORE_Y, jnp.asarray(state.score, jnp.int32), SCORE_SCALE, GOLD_COLOR
        )

        raster = _draw_number_right(
            raster, ROUND_RIGHT_X, ROUND_Y, jnp.asarray(state.cpu_score, jnp.int32), SCORE_SCALE, GOLD_COLOR
        )

        
        return raster
    
    
#environment

class JaxHangman(JaxEnvironment[HangmanState, HangmanObservation, HangmanInfo, HangmanConstants]):
    # Full ALE action set for Hangman (18 actions):
    # ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 
    #  'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: HangmanConstants = None):
        consts = consts or HangmanConstants()
        super().__init__(consts)
        self.renderer = HangmanRenderer()
        self.consts = consts

        self.obs_size = L_MAX + L_MAX + ALPHABET_SIZE + 3

        # Compute derived values from constants
        self.timed = 1 if str(consts.DIFFICULTY_MODE).upper() == "A" else 0
        self.timer_steps = int(consts.TIMER_SECONDS * consts.STEPS_PER_SECOND)

        self._rng_key = jrandom.PRNGKey(0)

        
    def seed(self, seed: int | chex.Array) -> chex.Array:
        if isinstance(seed, int):
            key = jrandom.PRNGKey(int(seed))
        else:
            arr = jnp.asarray(seed)
            key = arr if arr.shape == (2,) else jrandom.PRNGKey(int(jnp.uint32(jnp.sum(arr))))
        self._rng_key = key
        return key


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[HangmanObservation, HangmanState]:
        if key is None:
            key = self._rng_key

        key, word, length = _sample_word(key)

        self._rng_key = key
        # init round timer 
        time0 = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
        tmax  = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)


        state = HangmanState(
            key=key,
            word=word,
            length=length,
            mask=jnp.zeros((L_MAX,), dtype=jnp.int32),
            guessed=jnp.zeros((ALPHABET_SIZE,), dtype=jnp.int32),
            misses=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_MISSES, dtype=jnp.int32),
            cursor_idx=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            reward=jnp.array(0.0, dtype=jnp.float32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            round_no=jnp.array(1, dtype=jnp.int32),
            time_left_steps=time0,
            timer_max_steps=tmax,
            cpu_score=jnp.array(0, dtype=jnp.int32),
            last_commit=jnp.array(0, dtype=jnp.int32),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HangmanState, action: chex.Array) -> Tuple[HangmanObservation, HangmanState, float, bool, HangmanInfo]:
        # Translate compact agent action index to ALE console action
        action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        commit = _action_commit(action).astype(jnp.int32)
        delta  = _action_delta_cursor(action)
        delta  = jnp.where(state.step_counter % 6 == 0, delta, 0)
        def _new_round_from(s: HangmanState) -> HangmanState:
            key, word, length = _sample_word(s.key)
            time0 = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
            tmax  = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
            return HangmanState(
                key=key, word=word, length=length,
                mask=jnp.zeros((L_MAX,), dtype=jnp.int32),
                guessed=jnp.zeros((ALPHABET_SIZE,), dtype=jnp.int32),
                misses=jnp.array(0, dtype=jnp.int32),
                lives=jnp.array(self.consts.MAX_MISSES, dtype=jnp.int32),
                cursor_idx=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(False),                     
                reward=s.reward,                           
                step_counter=s.step_counter,
                score=s.score,                             
                round_no=s.round_no,                       
                time_left_steps=time0,
                timer_max_steps=tmax,
                cpu_score=s.cpu_score,                     
                last_commit=jnp.array(0, dtype=jnp.int32),
            )



        def _new_round(s: HangmanState) -> HangmanState:
            key, word, length = _sample_word(s.key)
            time0 = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
            tmax  = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)

            return HangmanState(
                key=key, word=word, length=length,
                mask=jnp.zeros((L_MAX,), dtype=jnp.int32),
                guessed=jnp.zeros((ALPHABET_SIZE,), dtype=jnp.int32),
                misses=jnp.array(0, dtype=jnp.int32),
                lives=jnp.array(self.consts.MAX_MISSES, dtype=jnp.int32),
                cursor_idx=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(False),
                reward=jnp.array(0.0, dtype=jnp.float32),
                step_counter=jnp.array(0, dtype=jnp.int32),
                score=s.score,
                round_no=s.round_no,
                time_left_steps=time0,
                timer_max_steps=tmax,
                cpu_score=s.cpu_score,
                last_commit=jnp.array(0, dtype=jnp.int32),
            )


        def _continue_round(s: HangmanState) -> HangmanState:
            #(skip guessed)
            cursor = _advance_cursor_skip_guessed(s.cursor_idx, delta, s.guessed)

            #tier tick every step
            t0 = s.time_left_steps
            t1 = jnp.where(self.timed == 1, jnp.maximum(t0 - 1, 0), t0)
            timed_out = jnp.logical_and(self.timed == 1, t1 == 0)

            idx = jnp.arange(L_MAX, dtype=jnp.int32)
            within = idx < s.length
            
            def on_commit(s2: HangmanState) -> HangmanState:
                already  = s2.guessed[cursor] == 1
                guessed  = s2.guessed.at[cursor].set(1)

                # reveal any matches at valid positions
                pos_hits = (s2.word == cursor).astype(jnp.int32) * within.astype(jnp.int32)
                any_hit  = jnp.any(pos_hits == 1)
                mask     = jnp.where(pos_hits.astype(bool), 1, s2.mask)

                # wrong guess
                wrong   = jnp.logical_and(jnp.logical_not(any_hit), jnp.logical_not(already))
                misses  = s2.misses + wrong.astype(jnp.int32)
                lives   = s2.lives  - wrong.astype(jnp.int32)

                #win
                n_revealed   = jnp.sum(jnp.where(within, mask, 0))
                all_revealed = (n_revealed == s2.length)

                #loss check
                lost = misses >= self.consts.MAX_MISSES

                #reveal
                mask_final = jnp.where(lost, jnp.where(within, 1, mask), mask)

                # reward
                step_reward = jnp.where(all_revealed, 1.0,
                                jnp.where(lost, -1.0, self.consts.STEP_PENALTY)).astype(jnp.float32)

                #bump counters
                won          = all_revealed.astype(jnp.int32)
                lost_i32     = lost.astype(jnp.int32)
                round_ended  = jnp.logical_or(all_revealed, lost).astype(jnp.int32)

                new_score    = (s2.score + won).astype(jnp.int32)
                cpu_new      = (s2.cpu_score + lost_i32).astype(jnp.int32)
                new_roundno  = (s2.round_no + round_ended).astype(jnp.int32)

                base = HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=mask_final, guessed=guessed, misses=misses, lives=lives,
                    cursor_idx=cursor,
                    done=jnp.array(False),                        
                    reward=step_reward,
                    step_counter=s2.step_counter + 1,
                    score=new_score,                              
                    round_no=new_roundno,                         
                    time_left_steps=jnp.array(
                        self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32
                    ),
                    cpu_score=cpu_new,                            
                    timer_max_steps=s2.timer_max_steps,
                    last_commit=commit,
                )

                return lax.cond(
                    (round_ended == 1),
                    _new_round_from,              
                    lambda s_: s_,                
                    base
                )

            def no_commit(s2: HangmanState) -> HangmanState:
                active    = (self.timed == 1)
                t0        = s2.time_left_steps
                t1        = jnp.where(active, jnp.maximum(t0 - 1, 0), t0)
                timed_out = jnp.logical_and(active, t1 == 0)
                add_miss  = jnp.where(timed_out, 1, 0).astype(jnp.int32)

                misses = s2.misses + add_miss
                lives  = s2.lives  - add_miss
                lost   = misses >= self.consts.MAX_MISSES

                idx = jnp.arange(L_MAX, dtype=jnp.int32)
                within = idx < s2.length
                mask_final = jnp.where(lost, jnp.where(within, 1, s2.mask), s2.mask)

                t_next = jnp.where(
                    timed_out,
                    jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32),
                    t1
                )

                cpu_new     = s2.cpu_score + jnp.where(lost, 1, 0)
                round_ended = lost.astype(jnp.int32)
                new_roundno = s2.round_no + round_ended

                base = HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=mask_final, guessed=s2.guessed, misses=misses, lives=lives,
                    cursor_idx=cursor,
                    done=jnp.array(False),                        # never signal done
                    reward=jnp.where(lost, jnp.array(-1.0, dtype=jnp.float32),
                                     jnp.array(self.consts.STEP_PENALTY, dtype=jnp.float32)),
                    step_counter=s2.step_counter + 1,
                    score=s2.score,
                    round_no=new_roundno,
                    time_left_steps=t_next,
                    cpu_score=cpu_new,
                    timer_max_steps=s2.timer_max_steps,
                    last_commit=commit,
                )

                return lax.cond(
                    (round_ended == 1),
                    _new_round_from,
                    lambda s_: s_,
                    base
                )

            return lax.cond(commit, on_commit, no_commit, s)

        def _freeze(_s):
            return HangmanState(
                key=_s.key, word=_s.word, length=_s.length,
                mask=_s.mask, guessed=_s.guessed, misses=_s.misses, lives=_s.lives,
                cursor_idx=_s.cursor_idx,
                done=_s.done,
                reward=jnp.array(0.0, dtype=jnp.float32),   
                step_counter=_s.step_counter,
                score=_s.score, round_no=_s.round_no,
                time_left_steps=_s.time_left_steps,
                cpu_score=_s.cpu_score,
                timer_max_steps=_s.timer_max_steps,
                last_commit=commit,                         
            )

        next_state = _continue_round(state)

        done = self._get_done(next_state)                 # stays False
        env_reward = self._get_env_reward(state, next_state)

        obs = self._get_observation(next_state)
        info = self._get_info(next_state)
        return obs, next_state, env_reward, done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "revealed": spaces.Box(low=0, high=PAD_TOKEN, shape=(L_MAX,), dtype=jnp.int32),
            "mask":     spaces.Box(low=0, high=1,          shape=(L_MAX,), dtype=jnp.int32),
            "guessed":  spaces.Box(low=0, high=1,          shape=(ALPHABET_SIZE,), dtype=jnp.int32),
            "misses":   spaces.Box(low=0, high=self.consts.MAX_MISSES, shape=(), dtype=jnp.int32),
            "lives":    spaces.Box(low=0, high=self.consts.MAX_MISSES, shape=(), dtype=jnp.int32),
            "cursor_idx": spaces.Box(low=0, high=ALPHABET_SIZE-1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=jnp.uint8)


    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HangmanState) -> HangmanObservation:
        revealed = _compute_revealed(state.word, state.mask)
        return HangmanObservation(
            revealed=revealed, mask=state.mask, guessed=state.guessed,
            misses=state.misses, lives=state.lives, cursor_idx=state.cursor_idx,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: HangmanObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.revealed.flatten(),
            obs.mask.flatten(),
            obs.guessed.flatten(),
            obs.misses.reshape((1,)).astype(jnp.int32),
            obs.lives.reshape((1,)).astype(jnp.int32),
            obs.cursor_idx.reshape((1,)).astype(jnp.int32),
        ])


    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: HangmanState, state: HangmanState):
        # single scalar float32
        return jnp.asarray(state.reward, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: HangmanState, state: HangmanState):
        #return the reward 
        return state.reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: HangmanState) -> bool:
        return state.done

    # @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HangmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_info(self, state: HangmanState) -> HangmanInfo:
        return HangmanInfo(time=state.step_counter)