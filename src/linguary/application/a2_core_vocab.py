"""Built-in CEFR A1-A2 core English vocabulary and deterministic lexical filter.

Pure application-layer helper with zero external dependencies. Used to detect
non-basic (B1+) words that slip into `READ_AS_IS` blocks during A2 calibration.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_APOSTROPHES = str.maketrans({
    "\u2019": "'",
    "\u2018": "'",
    "\u02bc": "'",
    "`": "'",
})

_TOKEN_RE = re.compile(
    r"[A-Za-z\u00C0-\u024F0-9]+(?:['\u2019\u2018\u02bc-][A-Za-z\u00C0-\u024F0-9]+)*"
)
_NUMBER_TOKEN_RE = re.compile(r"^\d+(?:st|nd|rd|th|s)?$", re.IGNORECASE)
_ROMAN_NUMERAL_RE = re.compile(
    r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$"
)
_TITLE_ABBREV_RE = re.compile(r"(?:^|\W)(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.$", re.IGNORECASE)
_INITIAL_PERIOD_RE = re.compile(r"(?:^|\W)[A-Z]\.$")
_MONOSYLLABLE_CVC_RE = re.compile(r"^[bcdfghjklmnpqrstvwxyz]*[aeiou][bcdfgmnprstz]$")

_DOUBLEABLE_CONSONANTS = frozenset("bcdfglmnprstz")
_ES_ENDINGS = ("shes", "ches", "xes", "zes", "sses", "oes")

_OPENING_IMMEDIATE = frozenset('"\u201c\u2018\u00ab([{')
_CLOSING_PUNCT = '"\u201d\u2019\'\u00bb)]}'
_SENTENCE_END_CHARS = frozenset(".!?\u2026:;")

#: Words that must NEVER be treated as basic A1-A2 core vocabulary, even if
#: suffix stripping would otherwise reduce them to a core root (e.g. `hardly` -> `hard`).
_NON_CORE_OVERRIDE: frozenset[str] = frozenset({
    "actually",
    "hardly",
    "barely",
    "merely",
    "scarcely",
    "likely",
    "unlikely",
    "mostly",
    "roughly",
    "survivor",
    "survivors",
    "bump",
    "bumped",
    "rip",
    "rips",
    "scratch",
    "scratches",
    "whoa",
    "wink",
    "winked",
    "groan",
    "groaned",
    "chuckle",
    "chuckled",
    "peer",
    "peered",
    "nod",
    "nodded",
    "snigger",
    "sniggered",
    "moan",
    "moaned",
    "uneasy",
    "nervous",
    "demand",
    "demanded",
    "murmur",
    "murmured",
    "snap",
    "snapped",
    "mutter",
    "muttered",
    "whisper",
    "whispered",
    "gasp",
    "gasped",
    "stammer",
    "stammered",
    "budge",
    "cannon",
    "sweetums",
    "sweetheart",
    "suppose",
    "freckle",
    "freckled",
    "wizard",
    "wizards",
    "afterlight",
    "afterlights",
    "skinjacker",
    "windshield",
    "collision",
    "highway",
    "eternity",
    "frown",
    "frowned",
    "shrug",
    "shrugged",
    "glare",
    "glared",
    "stare",
    "stared",
    "tremble",
    "trembled",
    "shiver",
    "shivered",
    "gulp",
    "gulped",
    "blink",
    "blinked",
    "sigh",
    "sighed",
    "grin",
    "grinned",
    "scowl",
    "scowled",
    "sneer",
    "sneered",
    "yell",
    "yelled",
    "scream",
    "screamed",
    "sob",
    "sobbed",
})

_SUFFIX_FALSE_POSITIVES: frozenset[str] = frozenset({
    "adder",
    "batter",
    "blister",
    "chatter",
    "clatter",
    "clutter",
    "flatter",
    "flutter",
    "fodder",
    "gutter",
    "patter",
    "rudder",
    "shatter",
    "shed",
    "shudder",
    "shutter",
    "spatter",
    "sputter",
    "stutter",
    "tatter",
    "wed",
})


def _words(block: str) -> frozenset[str]:
    return frozenset(block.split())


# Function words, pronouns, determiners, prepositions, conjunctions, auxiliaries, contractions.
_FUNCTION_WORDS: frozenset[str] = _words("""
a an the this that these those
i me my mine myself
you your yours yourself yourselves
he him his himself
she her hers herself
it its itself
we us our ours ourselves
they them their theirs themselves
who whom whose which what where when why how
whoever whatever whichever wherever whenever however
all any another both each either enough every few fewer fewest
less least little many more most much neither no none other others
several some such
anybody anyone anything anywhere
everybody everyone everything everywhere
nobody nothing nowhere
somebody someone something somewhere
about above across after against along among amongst around as at
before behind below beneath beside besides between beyond by
despite down during except for from in inside into like near next
of off on onto opposite out outside over past per round since
than through throughout till to toward towards under underneath
unlike until up upon via with within without
and although because but if nor or so though unless whereas whether while yet
am is are was were be been being
do does did done doing
have has had having
can could may might must ought shall should will would cannot
not never yes yeah yep nope nah ok okay
mr mrs ms miss dr prof st sir madam ma'am
o'clock
i'm i've i'll i'd
you're you've you'll you'd
he's he'll he'd
she's she'll she'd
it's it'll it'd
we're we've we'll we'd
they're they've they'll they'd
that's that'll that'd
there's there're there'll there'd
here's
what's what'll what're
who's who'll who'd who're who've
where's when's why's how's
isn't aren't wasn't weren't
don't doesn't didn't
haven't hasn't hadn't
won't wouldn't can't couldn't shouldn't mustn't mightn't needn't
let's
""")

# Irregular past / participle / plural / comparative forms not eligible for suffix derivation.
_IRREGULAR_ONLY_FORMS: frozenset[str] = _words("""
arose arisen ate awoke awoken
beaten became began begun bent bet bit bitten bled blew blown
bought broke broken brought built burnt
caught children chose chosen came crept
dealt dug drank drawn dreamt drew driven drove drunk
eaten
fallen fed fell felt feet flew flown forgot forgotten forgave forgiven fought found froze frozen
gave geese gone got gotten grew grown
heard held hid hidden hung
kept knew known knives
laid lain learnt leaves led lent lit lives loaves lost
made meant men met mice
paid people
ran rang ridden risen rose rung
said sang sank sat saw seen sent shaken shelves shone shook shot shown
shrank shrunk slept slid smelt sold spelt spoke spoken spent spilt spun spoilt
stood stole stolen struck stuck stung swam swept swore sworn swum swung
taken taught teeth thieves thought threw thrown told took tore torn
understood
went woke woken women won wore worn wrote written wives wolves
better best worse worst further furthest farther farthest elder eldest
halves scarves calves
""")

# Numbers, calendar, time, and basic measurements.
_NUMBERS_AND_TIME: frozenset[str] = _words("""
zero one two three four five six seven eight nine ten
eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen
twenty thirty forty fifty sixty seventy eighty ninety
hundred thousand million billion
first second third fourth fifth sixth seventh eighth ninth tenth
eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth
twentieth thirtieth fortieth fiftieth hundredth
once twice half quarter double single dozen pair couple
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
spring summer autumn fall winter
morning afternoon evening night midnight noon
today tonight tomorrow yesterday weekday weekend
day week month year decade century season date
hour minute moment time age period
birthday holiday vacation christmas easter
clock watch calendar schedule timetable alarm
meter metre kilometer kilometre centimeter centimetre millimeter millimetre
mile foot inch gram kilogram kilo pound liter litre degree
km cm mm kg pm
""")

# Basic interjections and discourse markers.
_INTERJECTIONS: frozenset[str] = _words("""
ah aha bye eh goodbye hello hey hi hm hmm hooray hurray
oh oops ouch pardon please shh shoo sorry thanks thank
ugh uh um welcome well wow yay bravo
""")

# Core CEFR A1-A2 verbs.
_VERBS: frozenset[str] = _words("""
accept act add admit advise afford agree aim allow announce annoy answer apologize
appear apply argue arrange arrive ask attack attend attract avoid
bake bang beat become beg begin behave believe belong bend bet bite bleed blow
boil book borrow bother bounce bow break breathe bring brush build burn burst
bury buy
calculate call calm camp cancel care carry catch cause celebrate change charge
chase chat cheat check cheer chew choose clap clean clear climb close coach
collect color colour comb combine come comfort command communicate compare compete
complain complete confirm confuse connect consider contact contain continue
control cook copy correct cost cough count cover crack crash crawl create cross
cry cure cut cycle
damage dance dare date deal decide decorate decrease defend delay deliver depend
describe deserve design destroy develop die dig direct disagree disappear
disappoint discover discuss dislike dive divide drag draw dream dress drink drive
drop drown dry
earn eat edit email employ empty encourage end enjoy enter entertain escape
examine exchange excite excuse exercise exist expect explain explore express
face fail fall fancy fasten fear feed feel fetch fight fill film find finish fire
fish fit fix flash float flood flow fly fold follow force forget forgive form
freeze frighten fry
gain gather get give go grab graduate greet grow guard guess guide
hand handle hang happen harm hate heal hear heat help hide hire hit hold hope hug
hunt hurry hurt
ignore imagine improve include increase inform injure insist install instruct
intend interest interrupt introduce invent invite involve iron
jog join joke judge jump
keep kick kill kiss knock know
label land last laugh lay lead lean learn leave lend let lie lift light like limit
link list listen live load lock look lose love lower
mail make manage march mark marry match matter mean measure meet melt mend
mention mess mind miss mix move
name need notice
offer open operate order organize organise owe own
pack paint park pass paste pat pause pay peel perform permit phone photograph
pick picnic place plan plant play please plug point polish pop pour practice
practise praise pray predict prefer prepare present press pretend prevent print
produce promise pronounce protect prove provide publish pull pump punch punish
push put
question queue quit
race rain raise reach react read realize realise receive recognize recognise
recommend record recover reduce refuse regret relax release remain remember
remind remove rent repair repeat replace reply report request rescue research
reserve respect respond rest result return review ride ring rise risk rob rock
roll row rub ruin rule run rush
sail save say scare score search seat see seem select sell send sense separate
serve set settle shake share shave shine shoot shop shout show shut sign signal
sing sink sit skate ski skip sleep slice slide slip slow smell smile smoke snow
solve sort sound speak spell spend spill spin split spoil spread stand start state
stay steal step stick stop store study succeed suffer suggest suit supply support
surf surprise surround switch
take talk tap taste teach tear telephone tell test text thank think throw tidy tie
tip tire touch tour trade train translate transport travel treat trick trip trust
try turn type
understand unpack update upgrade upload use
value visit volunteer vote
wait wake walk want warm warn wash waste watch water wave wear weigh welcome wet
win wind wipe wish wonder work worry wrap write
""")

# Core CEFR A1-A2 nouns (curated Oxford 3000 A1-A2 & Cambridge KET/PET core).
_NOUNS: frozenset[str] = _words("""
ability accident account ache act action activity actor actress ad addition
address adult advantage adventure advert advertisement advice afternoon age agent
agreement aid aim air airline airport alarm album alphabet amount anger angle
animal ankle answer apartment apology app apple appointment area argument arm
armchair army arrival art article artist assistant athlete atmosphere attack
attention attitude audience aunt author autumn avenue award
baby back background backpack bag baggage bakery balance balcony ball balloon
banana band bank bar base baseball basket basketball bat bath bathroom battery
beach bean bear beard beauty bed bedroom bee beef beer beginning bell belt bench
bicycle bike bill bin biology bird birth biscuit bit bite blanket block blog
blood board boat body bone book bookcase bookshelf bookshop bookstore boot
border boss bottle bottom bowl box boy boyfriend brain bread break breakfast
breath bridge brother brush bucket builder building burger burn bus bush
business businessman businesswoman butter butterfly button
cake calendar call camel camera camp campsite can candle candy cap capital
captain car card career carpet carrot carry cartoon case cash castle cat cause
cave ceiling cent center centre century chain chair challenge champion chance
change channel chapter character chat check cheek cheese chef chemistry chess
chest chicken chief child childhood chin chip chocolate choice church cinema
circle circus citizen city class classmate classroom cleaner climate clinic
clock clothes clothing cloud club coach coast coat code coffee coin cold
colleague collection college color colour comb comedy comfort comment
communication community company comparison competition complaint computer concert
conclusion condition confidence connection contact content contest continent
contract control conversation cook cookie cooking copy corner cost cottage
cotton cough counter country countryside couple course court cousin cover cow
cream credit crew crime crowd crown cry culture cup cupboard curtain cushion
customer cut cycle
dad daddy damage dance dancer danger dark data date daughter day deal death
decision degree delay delivery dentist department description desert design
designer desk dessert destination detail development device dialogue diary
dictionary diet difference difficulty dinner direction director dirt discount
discussion disease dish display distance doctor document dog doll dollar donkey
door dot doubt dozen drama drawer drawing dream dress drink driver drop drug
drum duck dust duty
eagle ear earth earthquake east edge edition editor education effect effort egg
elbow election electricity element elephant elevator email emergency emotion
employee employer employment end enemy energy engine engineer entrance entry
envelope environment episode equipment error escape essay euro eve evening event
exam examination example exception exchange excitement excuse exercise exhibition
exit expense experience experiment expert explanation expression extra eye
face fact factory failure fair fairy fall family fan fantasy farm farmer farming
fashion fat father fault favor favour fear feather feature fee feeling fellow
female fence ferry festival fever fiction field fight figure file film final
finance finger finish fire firefighter fireplace firm fish fisherman fishing fit
fitness flag flash flat flavor flavour flight float flood floor flour flow flower
flu flute fly focus fog food fool foot football force forehead forest fork form
format forward fox frame freedom freezer fridge friend friendship frog front frost
fruit fuel fun function fur furniture future
gadget galaxy gallery game gap garage garbage garden gardener garlic gas gate
general genius gentleman geography geometry ghost giant gift giraffe girl
girlfriend glass glasses globe glove glue goal goat god gold golf good goodbye
goods goose government grade graduate grammar grandad grandchild granddaughter
grandfather grandma grandmother grandpa grandparent grandson grape graph grass
grave gravity gray green greengrocer greenhouse greeting grey grill grocery ground
group growth guard guess guest guide guitar gun guy gym
habit hair haircut hairdresser half hall hallway ham hammer hand handbag
handkerchief handle handwriting happiness harbor harbour harm hat hate head
headache headline health hearing heart heat heater heating heaven heel height
helicopter hell helmet help helper hen hero high hill hint hip history hit hobby
hockey hold hole holiday home homework honey honeymoon honor honour hook hope
horror horse hospital host hostel hotel hour house housework hug human humor
humour hunger hunt hunter hurry hurt husband
ice idea ideal illness image imagination impact importance impression improvement
inch income increase independence individual info information ingredient injury
ink inn insect inside inspector instruction instructor instrument insurance
intelligence interest internet interview introduction invention inventor
invitation iron island item
jacket jam jar jazz jeans jewellery jewelry job joint joke journal journalist
journey joy judge juice jump jumper jungle
kettle key keyboard kick kid kilogram kilometer kilometre kind kindness king
kingdom kiss kit kitchen kite kitten knee knife knock knot knowledge
lab label ladder lady lake lamb lamp land lane language lap laptop last laugh
laundry law lawyer lead leader leaf leather leave left leg lemon lemonade length
lesson letter lettuce level library license licence lid life lift light lighter
lightning limit line link lion lip liquid list liter litre living load local
location lock log look lorry loss lot love luck luggage lunch lung
machine magazine magic mail mailbox main make makeup male mall man management
manager mango map marathon march mark market marriage mask mass master mat match
mate material math mathematics maths matter maximum mayor meal meaning measure
meat mechanic medal media medicine medium meeting melon member memory menu mess
message metal meter method metre microwave middle midnight might mile milk
million mind mine minimum minister minus minute mirror miss mist mistake mix
mobile model modern moment money monitor monkey monster month mood moon morning
most mother motor motorbike motorcycle motorway mountain mouse mouth move
movement movie mud mug murder muscle museum mushroom music musician mystery
nail name nap napkin nation nature navy neck necklace need needle neighbor
neighbour neighborhood neighbourhood nest net network news newspaper nice niece
night noise noodle noon north nose note notebook nothing notice noun novel
number nurse nut
object ocean odd offer office officer official oil onion online opening opera
operation opinion opposite option orange orchestra order organization
organisation origin original outside oven overcoat owl owner
pack package packet page pain paint painter painting pair pajamas palace pan
pancake panda panic pants paper pardon parent park parking parrot part partner
party pass passage passenger passport password past pasta pat patch path patient
pattern pause pay pea peace peach peanut pear pen pencil penguin penny people
pepper percent perfume period permission permit person pet petrol pharmacy phone
photo photograph photographer photography phrase physics piano picnic picture pie
piece pig pigeon pile pill pillow pilot pin pine pineapple pink pipe pirate pizza
place plain plan plane planet planning plant plastic plate platform play player
playground pleasure plenty plug plum plumber plus pocket podcast poem poet point
polar pole police policeman policewoman polish pond pony pool pop popcorn pork
port position post postcard poster postman pot potato pound powder power practice
practise present president press price pride priest prince princess print printer
prison prisoner private prize problem product profession professional professor
profile program programme programmer progress project promise proof pub public
pull pullover pump pumpkin punch pupil puppy purple purpose purse push puzzle
pyjamas pyramid
quality quarter queen question queue quiet quiz
rabbit race racing rack radio rail railroad railway rain rainbow raincoat range
rat rate raw ray reach reader reading reason recipe record recorder recording red
refrigerator region regret relation relative relaxation release relief religion
remark reminder remote rent repair repeat reply report reporter request rescue
research reservation reserve respect response rest restaurant result return
review ride rider right ring rise risk river roast robot rock rocket role roll
romance roof room root rope rose round route routine row royal rubber rude rug
rule ruler run runner runway rush
sadness safe safety sail sailor salad sale salon salt sand sandal sandwich sauce
saucer sausage saving saw scare scarf scene schedule school science scientist
scissors scooter score screen script sea seafood seal search season seat seaweed
second secret secretary section seed self sense sentence series server service
set shade shadow shake shampoo shape share shark shave sheep sheet shelf shell
shelter shield shine ship shirt shock shoe shoot shop shopper shopping shore
short shorts shoulder shout show shower shut side sight sign signal silence silk
silver simple singer singing single sink sir sister site situation size skate
skater ski skier skill skin skirt sky sleep sleeper sleeve slide slip slipper
smart smell smile smoke smoker snack snake sneaker snow snowball snowman soap
soccer sock soda sofa software soldier solo son song soon sort soul sound soup
sour source south souvenir space spacecraft spaceship speaker special speed
spell spelling spider spill spinach spirit sport spot spray spread spring square
squirrel stadium stage stair staircase stamp stand star start state station
statue stay steak steam steel step stereo stick sticker still sting stomach stone
stool stop storage store storm story stove stranger strap straw strawberry stream
street strength stretch strike string strip stripe strong student studio study
stuff style subject subway success sugar suggestion suit suitcase suite sum
summer sun sunflower sunglasses sunlight sunrise sunset sunshine supermarket
supper supply support supporter surface surname surprise sweet swim swimmer
swimming swing switch
table tablespoon tablet tag tail tailor tale talent talk tank tap tape target task
taste tax taxi tea teacher teaching team tear teaspoon technical technician
technology teen teenager teeth telephone telescope television temperature temple
tennis tent term test text textbook theater theatre theme thermometer thief thing
thinking thirst thought thread threat throat thumb thunder tick ticket tide tie
tiger tile time timetable tin tip tire title toast today toe toilet tomato tomorrow
ton tone tongue tonight tool tooth toothbrush toothpaste top topic torch total
touch tour tourist towel tower town toy track trade traffic train trainer
training tram transport travel traveler traveller tray treasure treat tree trip
truck true trunk trust truth try tube tulip tummy tuna tune tunnel turkey turn
turtle tutor tv twin twist type
umbrella uncle underwear unit university unknown upper upset urban use user
vacation valley van vanilla variety vase vegetable vehicle version vest vet
victory video view viewer village violin virus visa vision visit visitor vitamin
voice volleyball volume volunteer vote vowel
wait waiter waitress wake walk walker wall wallet want war wardrobe warmth
warning wash washer washing waste watch water waterfall watermelon wave way
weather web website wedding week weekday weekend weight welcome well west whale
wheel white whole wife wild will win wind window wine wing winner winter wipe
wire wish wolf woman wonder wood wool word work worker workshop world worm
worry wrap wrist writer writing wrong
yard year yellow yesterday yoga yogurt yoghurt young
zebra zero zip zipper zone zoo
""")

# Core CEFR A1-A2 adjectives (eligible for -er, -est, -ly derivation).
_ADJECTIVES: frozenset[str] = _words("""
able afraid african alive alone aloud amazing american ancient angry annual arab
arabic asian asleep attractive automatic available average awake awful
bad bald basic beautiful beige best better big bitter black blank blind blond
blonde blue boiling boring born bottom brave brief bright brilliant british
broken brown busy
calm canadian careful careless central certain cheap cheerful chinese clean clear
clever close closed cloudy cold colorful colourful comfortable common complete
confident confused confusing convenient cool correct crazy creative crowded
cultural curious current cute
daily damaged dangerous dark dead deaf dear deep definite delicious delighted
different difficult digital direct dirty disabled disappointed dishonest distant
double downtown dressed dried drunk dry dull dumb dutch
early eastern easy economic educational electric electrical electronic elementary
emotional empty energetic english enormous equal european even everyday exact
excellent excited exciting expensive experienced extra
fair false familiar famous fantastic far fashionable fast fat favorite favourite
female few final financial fine firm first fit fixed flat flying foggy foreign
formal former fortunate forward free freezing french frequent fresh friendly
frightened frightening front frozen full fun funny further future
general gentle german giant glad global golden good grand great green grey gray
handsome handy happy hard healthy heavy helpful hidden high historic historical
honest hopeful horrible hot huge human hungry
icy ideal ill immediate important impossible indoor informal injured inside
intelligent interested interesting international irish iron italian
japanese jealous joyful junior just
key kind known korean
large last late latin lazy leading left less level light limited liquid little
live lively living local lonely long loose lost loud lovely low lucky
mad magic magical main major male manual many married maximum mean medical medium
messy metal mexican middle mild minimal minimum minus missing mixed mobile modern
monthly more most mother motor muddy musical
narrow national natural nearby neat necessary negative new next nice noisy normal
northern
obvious odd official old olympic one online only open opposite orange ordinary
original other outdoor outside own
painful pale past patient peaceful perfect personal physical pink plain plastic
pleasant pleased polite political poor popular positive possible powerful
practical present pretty previous private professional proud public pure purple
quiet quick
rainy rapid rare raw ready real recent red regular relaxed rich right round royal
rubber rude russian
sad safe salty same sandy satisfied scary scottish second secret separate serious
sharp shiny shocked shocking short shy sick silent silly silver similar simple
single sleepy slow small smart smelly smooth snowy social soft solo sorry sound
southern spanish spare special spicy spoken sporting square standard steep sticky
still straight strange strong stuck stupid stylish successful sudden sunny super
sure surprised surprising sweet
tall tasty technical teen terrible thick thin thirsty tidy tight tiny tired tiring
top total true typical
ugly uncomfortable underground unhappy uniform united unknown unpleasant unusual
upset upstairs useful useless usual
various vegetarian very warm weak weekly welcome well western wet white whole wide
wild windy wise wonderful wooden woollen woolen working worried worse worst worth
wrong
yellow young
""")

# Core CEFR A1-A2 adverbs (`actually`, `hardly`, `barely`, `merely`, `scarcely`,
# `likely`, `unlikely`, `mostly`, `roughly` are intentionally excluded; regular
# `-ly` adverbs derived from `_ADJECTIVES` are matched via `_matches_ly`).
_ADVERBS: frozenset[str] = _words("""
abroad again ago ahead almost alone already also always anymore anyway anywhere
apart around aside away
back backward backwards before behind below beside besides best better between
downstairs early else elsewhere enough especially even eventually ever everywhere
far fast finally forever forward forwards
gradually here however
indoors inside instead just
lately later least less literally
mainly maybe meanwhile more
near nearby nearly never next now nowhere
off often once online only otherwise out outdoors outside over overnight overseas
partly perhaps pretty probably quite
rarely rather really right
seldom since somehow sometimes somewhat somewhere soon south still straight
then there therefore today together tomorrow tonight too truly twice
upstairs upward upwards usually
very
well west worldwide worse worst
yesterday yet
""")

# Transparent internationalisms commonly understood even at A1-A2.
_INTERNATIONALISMS: frozenset[str] = _words("""
academy actor actress album ambulance app artist astronaut auto
banana bank bar battery biology blog bus
cafe calendar camera camp captain center centre champion channel character chef
chemistry chocolate cinema circus classic client clinic club coach cocoa code
coffee comedy company computer concert contact continent control costume credit
culture
design designer dessert detective dialogue diamond diet digital director doctor
document dollar drama dragon
economic economy effect electric electronic elephant email energy engineer euro
exam express extra
factory family fan fantasy fashion festival film final finance firm fitness flag
flash focus football format
galaxy garage gas general genius geography global golf gram group guide guitar gym
helicopter hero history hobby hockey hospital hostel hotel
idea ideal image individual industry info information interest international
internet interview
jacket jazz jeans journal journalist juice
ketchup kilogram kilometer kilometre kiwi
lamp leader lemon lemonade limit line lion liter litre local
machine magic mango market mask master material math mathematics maths maximum
medal media medical medicine melon menu metal meter method metre metro mini
minimum minute mobile model modern moment monster motor museum music musician
mystery
national natural negative normal note
object ocean office officer official orange orchestra original
palace panda park partner party pass passport pasta pause percent period personal
phone photo photograph photographer physics piano picnic pilot pizza plan planet
plastic platform police pop popular port positive poster practical practice
president press print printer private prize problem process product profession
professional professor profile program programme project public
radio real record regular report reporter restaurant result risk robot rocket role
romance route routine
salad sandwich sausage scene school score secret secretary series serious server
service shampoo shock show signal simple site situation sofa soup special sport
stadium stage standard star start station stop student studio style super
supermarket surprise system
tablet talent taxi team technical technology telephone telescope television tennis
test text theater theatre ticket tiger title toast tomato ton tone topic total
tour tourist traffic train trainer tram transport tunnel type
umbrella uniform unit university
vacation video virus visa vision visit vitamin volleyball
wifi
yogurt yoghurt
zebra zero zone zoo
""")

# Strictly closed-class words that never take regular inflectional suffixes.
_NON_INFLECTABLE: frozenset[str] = _FUNCTION_WORDS - (_VERBS | _NOUNS | _ADJECTIVES)

_GRADABLE_ADVERBS: frozenset[str] = _words("""
close deep early far fast hard high late long loud low near quick slow soft soon
""")

_LY_BASE_WORDS: frozenset[str] = _words("""
day friend hour love month near part week year
""")

_PLURAL_OR_3SG_ROOTS: frozenset[str] = (
    (_NOUNS | _VERBS | _ADJECTIVES | _NUMBERS_AND_TIME | _INTERNATIONALISMS)
    - _NON_INFLECTABLE
    - _IRREGULAR_ONLY_FORMS
    - _NON_CORE_OVERRIDE
)

_VERB_INFLECT_ROOTS: frozenset[str] = (
    (_VERBS | _NOUNS | _ADJECTIVES | _INTERNATIONALISMS)
    - _NON_INFLECTABLE
    - _IRREGULAR_ONLY_FORMS
    - _NON_CORE_OVERRIDE
)

_ER_ROOTS: frozenset[str] = (
    (_ADJECTIVES | _VERBS | _GRADABLE_ADVERBS)
    - _NON_INFLECTABLE
    - _IRREGULAR_ONLY_FORMS
    - _NON_CORE_OVERRIDE
)

_EST_ROOTS: frozenset[str] = (
    (_ADJECTIVES | _GRADABLE_ADVERBS)
    - _NON_INFLECTABLE
    - _IRREGULAR_ONLY_FORMS
    - _NON_CORE_OVERRIDE
)

_LY_ROOTS: frozenset[str] = (
    (_ADJECTIVES | _LY_BASE_WORDS)
    - _NON_INFLECTABLE
    - _IRREGULAR_ONLY_FORMS
    - _NON_CORE_OVERRIDE
)

#: Built-in CEFR A1-A2 core English vocabulary (~2,500-3,000 normalized entries).
A2_CORE_VOCABULARY: frozenset[str] = (
    _FUNCTION_WORDS
    | _IRREGULAR_ONLY_FORMS
    | _NUMBERS_AND_TIME
    | _INTERJECTIONS
    | _VERBS
    | _NOUNS
    | _ADJECTIVES
    | _ADVERBS
    | _INTERNATIONALISMS
) - _NON_CORE_OVERRIDE


def _normalize_token(word: str) -> str:
    return word.strip().translate(_APOSTROPHES).lower().strip("'")


def _allows_undoubled_stem(stem: str) -> bool:
    return not _MONOSYLLABLE_CVC_RE.fullmatch(stem)


def _matches_plural_or_3sg(word: str) -> bool:
    if len(word) < 3 or not word.endswith("s"):
        return False
    if word.endswith("ies") and len(word) >= 4:
        if (word[:-3] + "y") in _PLURAL_OR_3SG_ROOTS or word[:-1] in _PLURAL_OR_3SG_ROOTS:
            return True
    if word.endswith("ves") and len(word) >= 5:
        if (word[:-3] + "f") in _PLURAL_OR_3SG_ROOTS or (word[:-3] + "fe") in _PLURAL_OR_3SG_ROOTS:
            return True
    if word.endswith(_ES_ENDINGS) and len(word) >= 4:
        stem = word[:-2]
        if stem in _PLURAL_OR_3SG_ROOTS:
            return True
        if (
            len(stem) >= 4
            and stem[-1] == stem[-2]
            and stem[-1] in _DOUBLEABLE_CONSONANTS
            and stem[:-1] in _PLURAL_OR_3SG_ROOTS
        ):
            return True
    if not word.endswith(("ss", "us", "is")) and word[:-1] in _PLURAL_OR_3SG_ROOTS:
        return True
    return False


def _matches_ed(word: str) -> bool:
    if len(word) < 4 or not word.endswith("ed"):
        return False
    if word.endswith("ied") and len(word) >= 4:
        if (word[:-3] + "y") in _VERB_INFLECT_ROOTS or word[:-1] in _VERB_INFLECT_ROOTS:
            return True
    stem = word[:-2]
    if len(stem) >= 3 and _allows_undoubled_stem(stem) and stem in _VERB_INFLECT_ROOTS:
        return True
    if len(word) >= 4 and word[:-1] in _VERB_INFLECT_ROOTS:
        return True
    if (
        len(stem) >= 4
        and stem[-1] == stem[-2]
        and stem[-1] in _DOUBLEABLE_CONSONANTS
        and stem[:-1] in _VERB_INFLECT_ROOTS
    ):
        return True
    return False


def _matches_ing(word: str) -> bool:
    if len(word) < 5 or not word.endswith("ing"):
        return False
    if word.endswith("ying") and (word[:-4] + "ie") in _VERB_INFLECT_ROOTS:
        return True
    stem = word[:-3]
    if len(stem) >= 3 and _allows_undoubled_stem(stem) and stem in _VERB_INFLECT_ROOTS:
        return True
    if len(stem) >= 2 and (stem + "e") in _VERB_INFLECT_ROOTS:
        return True
    if (
        len(stem) >= 4
        and stem[-1] == stem[-2]
        and stem[-1] in _DOUBLEABLE_CONSONANTS
        and stem[:-1] in _VERB_INFLECT_ROOTS
    ):
        return True
    return False


def _matches_er(word: str) -> bool:
    if len(word) < 4 or not word.endswith("er"):
        return False
    if word.endswith("ier") and len(word) >= 5 and (word[:-3] + "y") in _ER_ROOTS:
        return True
    stem = word[:-2]
    if len(stem) >= 3 and _allows_undoubled_stem(stem) and stem in _ER_ROOTS:
        return True
    if len(stem) >= 2 and (stem + "e") in _ER_ROOTS:
        return True
    if (
        len(stem) >= 4
        and stem[-1] == stem[-2]
        and stem[-1] in _DOUBLEABLE_CONSONANTS
        and stem[:-1] in _ER_ROOTS
    ):
        return True
    return False


def _matches_est(word: str) -> bool:
    if len(word) < 5 or not word.endswith("est"):
        return False
    if word.endswith("iest") and len(word) >= 6 and (word[:-4] + "y") in _EST_ROOTS:
        return True
    stem = word[:-3]
    if len(stem) >= 3 and _allows_undoubled_stem(stem) and stem in _EST_ROOTS:
        return True
    if len(stem) >= 3 and (stem + "e") in _EST_ROOTS:
        return True
    if (
        len(stem) >= 4
        and stem[-1] == stem[-2]
        and stem[-1] in _DOUBLEABLE_CONSONANTS
        and stem[:-1] in _EST_ROOTS
    ):
        return True
    return False


def _matches_ly(word: str) -> bool:
    if len(word) < 5 or not word.endswith("ly"):
        return False
    if word.endswith("ily") and len(word) >= 6 and (word[:-3] + "y") in _LY_ROOTS:
        return True
    if word.endswith("ally") and len(word) >= 7:
        if word[:-2] in _LY_ROOTS or word[:-4] in _LY_ROOTS:
            return True
    if word[:-2] in _LY_ROOTS:
        return True
    if word.endswith(("bly", "ply", "tly", "dly", "gly", "kly")) and (word[:-1] + "e") in _LY_ROOTS:
        return True
    if word.endswith("lly") and word[:-1] in _LY_ROOTS:
        return True
    return False


def _is_core_normalized(norm: str) -> bool:
    if not norm or norm in _NON_CORE_OVERRIDE:
        return False
    if norm in A2_CORE_VOCABULARY:
        return True
    if norm.endswith("'s") and len(norm) > 2:
        base = norm[:-2]
        if base in _NON_CORE_OVERRIDE:
            return False
        return _is_core_normalized(base)
    if "'" in norm or not norm.isascii() or not norm.isalpha():
        return False
    if norm in _SUFFIX_FALSE_POSITIVES:
        return False
    return (
        _matches_plural_or_3sg(norm)
        or _matches_ed(norm)
        or _matches_ing(norm)
        or _matches_er(norm)
        or _matches_est(norm)
        or _matches_ly(norm)
    )


def is_core_a2_word(word: str) -> bool:
    """Return True if `word` belongs to the built-in CEFR A1-A2 core vocabulary."""
    return _is_core_normalized(_normalize_token(word))


def _known_name_tokens(known_names: Sequence[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for name in known_names:
        for match in _TOKEN_RE.finditer(name):
            raw = match.group(0)
            norm = _normalize_token(raw)
            if not norm:
                continue
            tokens.add(norm)
            if norm.endswith("'s") and len(norm) > 2:
                tokens.add(norm[:-2])
            if "-" in norm:
                for part in norm.split("-"):
                    clean = part.strip("'")
                    if clean:
                        tokens.add(clean)
                        if clean.endswith("'s") and len(clean) > 2:
                            tokens.add(clean[:-2])
    return frozenset(tokens)


def _is_known_name(norm: str, known_tokens: frozenset[str]) -> bool:
    if norm in known_tokens:
        return True
    if norm.endswith("'s") and len(norm) > 2 and norm[:-2] in known_tokens:
        return True
    return False


def _is_trivial_token(raw: str, norm: str) -> bool:
    if not norm or len(norm) <= 1:
        return True
    if norm.endswith("'s") and len(norm[:-2]) <= 1:
        return True
    if _NUMBER_TOKEN_RE.fullmatch(norm):
        return True
    if raw.isupper() and bool(_ROMAN_NUMERAL_RE.fullmatch(raw)):
        return True
    return False


def _is_sentence_or_speech_start(text: str, start: int) -> bool:
    if start <= 0:
        return True
    prefix = text[:start]
    if prefix[-1] in _OPENING_IMMEDIATE:
        return True
    if prefix[-1] == "'" and (len(prefix) == 1 or not prefix[-2].isalnum()):
        return True
    trimmed = prefix.rstrip()
    if not trimmed:
        return True
    if "\n" in prefix[len(trimmed):]:
        return True
    if trimmed[-1] in "\u00ab\u2014\u2013":
        return True
    unquoted = trimmed.rstrip(_CLOSING_PUNCT)
    if not unquoted:
        return True
    last_char = unquoted[-1]
    if last_char not in _SENTENCE_END_CHARS:
        return False
    if last_char == ".":
        if _TITLE_ABBREV_RE.search(unquoted) or _INITIAL_PERIOD_RE.search(unquoted):
            return False
    return True


def _canonical_reported_word(norm: str) -> str:
    if norm.endswith("'s") and len(norm) > 2:
        return norm[:-2]
    return norm


def non_core_a2_words(
    text: str,
    *,
    known_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return unique words in `text` outside the CEFR A1-A2 core, in order of appearance.

    Skips numbers, uppercase Roman numerals, single-character letters, tokens from
    `known_names`, and capitalized tokens not starting a sentence or direct speech.
    Hyphenated compounds are checked part by part.
    """
    known_tokens = _known_name_tokens(known_names)
    found: list[str] = []
    seen: set[str] = set()

    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        norm = _normalize_token(raw)
        if _is_trivial_token(raw, norm):
            continue
        if _is_known_name(norm, known_tokens):
            continue
        if raw[0].isupper() and not _is_sentence_or_speech_start(text, match.start()):
            continue

        if "-" in norm:
            if norm in _NON_CORE_OVERRIDE:
                reported = _canonical_reported_word(norm)
                if reported not in seen:
                    seen.add(reported)
                    found.append(reported)
                continue
            if norm in A2_CORE_VOCABULARY:
                continue
            raw_parts = raw.split("-")
            all_parts_core = True
            for raw_part in raw_parts:
                part_norm = _normalize_token(raw_part)
                if _is_trivial_token(raw_part, part_norm):
                    continue
                if _is_known_name(part_norm, known_tokens):
                    continue
                if not _is_core_normalized(part_norm):
                    all_parts_core = False
                    break
            if not all_parts_core:
                reported = _canonical_reported_word(norm)
                if reported not in seen:
                    seen.add(reported)
                    found.append(reported)
            continue

        if not _is_core_normalized(norm):
            reported = _canonical_reported_word(norm)
            if reported not in seen:
                seen.add(reported)
                found.append(reported)

    return tuple(found)
