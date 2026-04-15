"""
Priya Project - spaCy NER Training Script (Fixed)
Offsets are computed programmatically - no hardcoded char positions
Run from backend/ directory with venv activated:
    python train_ner.py
"""

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
import os
import subprocess
import sys
import random


def make_example(text, entities):
    anns = []
    for phrase, label in entities:
        start = text.find(phrase)
        if start == -1:
            print(f"  [WARN] '{phrase}' not found in: {text}")
            continue
        end = start + len(phrase)
        anns.append((start, end, label))
    return (text, {"entities": anns})


LOC = "CAMPUS_LOCATION"
FAC = "FACILITY_TYPE"
TIM = "TIME_REFERENCE"
EVT = "EVENT_TYPE"

RAW = [
    ("Where is the Northumbria Library?", [("Northumbria Library", LOC)]),
    ("How do I get to City Campus?", [("City Campus", LOC)]),
    ("Where is the Student Union?", [("Student Union", LOC)]),
    ("I need to find Ellison Building", [("Ellison Building", LOC)]),
    ("Where is the Sutherland Building?", [("Sutherland Building", LOC)]),
    ("Directions to Coach Lane Campus please", [("Coach Lane Campus", LOC)]),
    ("How far is City Campus East from here?", [("City Campus East", LOC)]),
    ("Where is the Camden Building?", [("Camden Building", LOC)]),
    ("I am looking for the Lipman Building", [("Lipman Building", LOC)]),
    ("Can you help me find the Squires Building?", [("Squires Building", LOC)]),
    ("Where is the Student Services Reception?", [("Student Services Reception", LOC)]),
    ("How do I get to the Sports Centre?", [("Sports Centre", LOC)]),
    ("Where is the main library on City Campus?", [("City Campus", LOC)]),
    ("I need directions to Ellison Building please", [("Ellison Building", LOC)]),
    ("Where exactly is the Sutherland Building located?", [("Sutherland Building", LOC)]),
    ("Can you tell me where the Student Union is?", [("Student Union", LOC)]),
    ("I am trying to find City Campus North", [("City Campus North", LOC)]),
    ("Where is Northumbria University City Campus?", [("Northumbria University City Campus", LOC)]),
    ("How do I reach the Camden Building from here?", [("Camden Building", LOC)]),
    ("Where is the Squires Building on campus?", [("Squires Building", LOC)]),
    ("What floor is Student Services on in Ellison Building?", [("Student Services", LOC), ("Ellison Building", LOC)]),
    ("Is the Lipman Building near City Campus East?", [("Lipman Building", LOC), ("City Campus East", LOC)]),
    ("Where is the Pandon Building?", [("Pandon Building", LOC)]),
    ("How far is Coach Lane from City Campus?", [("Coach Lane", LOC), ("City Campus", LOC)]),
    ("Find me the main entrance to City Campus", [("City Campus", LOC)]),
    ("I need the Northumbria Library address", [("Northumbria Library", LOC)]),
    ("Is there a shortcut to the Student Union from Ellison Building?", [("Student Union", LOC), ("Ellison Building", LOC)]),
    ("Where is the Graduate School office?", [("Graduate School", LOC)]),
    ("How do I get to the Law School building?", [("Law School", LOC)]),
    ("Where is the main reception in Ellison Building?", [("Ellison Building", LOC)]),
    ("Can I walk from City Campus to Coach Lane Campus?", [("City Campus", LOC), ("Coach Lane Campus", LOC)]),
    ("I need to get to the Sutherland Building quickly", [("Sutherland Building", LOC)]),
    ("What bus stop is closest to City Campus East?", [("City Campus East", LOC)]),
    ("How long does it take to walk from City Campus to Coach Lane?", [("City Campus", LOC), ("Coach Lane", LOC)]),
    ("Where is the Engineering block on campus?", [("Engineering block", LOC)]),
    ("Where is the nearest cafe on campus?", [("cafe", FAC)]),
    ("Is there a gym on campus?", [("gym", FAC)]),
    ("Where can I find a printer?", [("printer", FAC)]),
    ("Is there a coffee shop nearby?", [("coffee shop", FAC)]),
    ("Where is the nearest toilet block?", [("toilet", FAC)]),
    ("I need to find a vending machine", [("vending machine", FAC)]),
    ("Where is the student canteen?", [("canteen", FAC)]),
    ("Is there a pharmacy on campus?", [("pharmacy", FAC)]),
    ("Where can I find a study room?", [("study room", FAC)]),
    ("Is there a car park nearby?", [("car park", FAC)]),
    ("Where is the nearest ATM on campus?", [("ATM", FAC)]),
    ("Where is the printing room?", [("printing room", FAC)]),
    ("I need a quiet study space", [("quiet study space", FAC)]),
    ("Where is the nearest restaurant?", [("restaurant", FAC)]),
    ("Is there a bike shed on campus?", [("bike shed", FAC)]),
    ("Where can I find a locker?", [("locker", FAC)]),
    ("Where is the computer lab?", [("computer lab", FAC)]),
    ("I need to use a photocopier", [("photocopier", FAC)]),
    ("Where is the nearest water fountain?", [("water fountain", FAC)]),
    ("Is there a meditation room on campus?", [("meditation room", FAC)]),
    ("Where can I find the prayer room?", [("prayer room", FAC)]),
    ("I am looking for the first aid room", [("first aid room", FAC)]),
    ("Where is the shop on campus?", [("shop", FAC)]),
    ("Where is the student lounge?", [("student lounge", FAC)]),
    ("Is there a food court on City Campus?", [("food court", FAC), ("City Campus", LOC)]),
    ("Where is the disabled toilet in this building?", [("disabled toilet", FAC)]),
    ("Is there a swimming pool at Northumbria?", [("swimming pool", FAC)]),
    ("I need a quiet room to make a phone call", [("quiet room", FAC)]),
    ("Is there a nursery at Northumbria University?", [("nursery", FAC)]),
    ("Is there a canteen in the Ellison Building?", [("canteen", FAC), ("Ellison Building", LOC)]),
    ("Where is the student kitchen on campus?", [("student kitchen", FAC)]),
    ("Is there a shower room at the Sports Centre?", [("shower room", FAC), ("Sports Centre", LOC)]),
    ("Where is the nearest printing facility?", [("printing facility", FAC)]),
    ("Is the library open today?", [("today", TIM)]),
    ("Is the gym open on Saturday?", [("Saturday", TIM)]),
    ("What are the opening hours on Sunday?", [("Sunday", TIM)]),
    ("Is the Student Union open tomorrow?", [("tomorrow", TIM)]),
    ("What time does the library open on Monday?", [("Monday", TIM)]),
    ("Is the canteen open after 5pm?", [("after 5pm", TIM)]),
    ("What are the weekend hours for the sports centre?", [("weekend", TIM)]),
    ("Is the printing room open this evening?", [("this evening", TIM)]),
    ("Does the gym close early on Fridays?", [("Fridays", TIM)]),
    ("Is the library open during Christmas break?", [("Christmas break", TIM)]),
    ("Is Student Services open in the morning?", [("morning", TIM)]),
    ("Can I access the computer lab late at night?", [("late at night", TIM)]),
    ("Is the cafe open before 9am?", [("before 9am", TIM)]),
    ("What time does the gym open on weekdays?", [("weekdays", TIM)]),
    ("Is the library open on Easter Sunday?", [("Easter Sunday", TIM)]),
    ("What are the hours during exam period?", [("exam period", TIM)]),
    ("Is the canteen open at lunchtime?", [("lunchtime", TIM)]),
    ("Does the library open early on Wednesday?", [("Wednesday", TIM)]),
    ("Is there 24 hour access to the study rooms?", [("24 hour", TIM)]),
    ("Is the sports centre open on public holidays?", [("public holidays", TIM)]),
    ("What time does the union bar close on Thursday?", [("Thursday", TIM)]),
    ("Is the cafe open during reading week?", [("reading week", TIM)]),
    ("What are the late night library hours?", [("late night", TIM)]),
    ("Is Student Services open on Tuesday afternoon?", [("Tuesday", TIM), ("afternoon", TIM)]),
    ("Can I use the gym before 8am?", [("before 8am", TIM)]),
    ("Is the printing room open over the summer?", [("summer", TIM)]),
    ("What are the hours on New Year's Day?", [("New Year's Day", TIM)]),
    ("Is the canteen open in the afternoon?", [("afternoon", TIM)]),
    ("What time does the library close on Friday evening?", [("Friday", TIM), ("evening", TIM)]),
    ("Is the gym open on Bank Holiday Monday?", [("Bank Holiday Monday", TIM)]),
    ("Are the computer labs open overnight?", [("overnight", TIM)]),
    ("What are the opening times on Christmas Eve?", [("Christmas Eve", TIM)]),
    ("Is the cafe open on weekday mornings?", [("weekday mornings", TIM)]),
    ("Does the library stay open until midnight?", [("midnight", TIM)]),
    ("Is the sports centre open early on Sunday?", [("Sunday", TIM)]),
    ("When is the next careers fair?", [("careers fair", EVT)]),
    ("Is there an open day this month?", [("open day", EVT)]),
    ("Where is the graduation ceremony?", [("graduation ceremony", EVT)]),
    ("When is freshers week?", [("freshers week", EVT)]),
    ("Is there a research seminar today?", [("research seminar", EVT)]),
    ("When is the next student union event?", [("student union event", EVT)]),
    ("Where is the degree show being held?", [("degree show", EVT)]),
    ("Is there a job fair on campus this week?", [("job fair", EVT)]),
    ("When is the welcome week for new students?", [("welcome week", EVT)]),
    ("Is there a workshop on dissertation writing?", [("workshop", EVT)]),
    ("Where is the alumni event taking place?", [("alumni event", EVT)]),
    ("When is the next lecture for computing?", [("lecture", EVT)]),
    ("Where is the thesis submission event?", [("thesis submission event", EVT)]),
    ("Is there a student society fair this term?", [("student society fair", EVT)]),
    ("Where is the Christmas party being held?", [("Christmas party", EVT)]),
    ("When is the postgraduate open evening?", [("postgraduate open evening", EVT)]),
    ("Is there a hackathon on campus this weekend?", [("hackathon", EVT)]),
    ("Where is the business networking event?", [("business networking event", EVT)]),
    ("Is there a coding bootcamp this semester?", [("coding bootcamp", EVT)]),
    ("Where is the arts exhibition being held?", [("arts exhibition", EVT)]),
    ("When is the student awards ceremony?", [("student awards ceremony", EVT)]),
    ("Is there a volunteering fair this month?", [("volunteering fair", EVT)]),
    ("Where is the sports day event?", [("sports day", EVT)]),
    ("When is the next pub quiz at the Student Union?", [("pub quiz", EVT), ("Student Union", LOC)]),
    ("Is there an exam revision workshop this week?", [("exam revision workshop", EVT)]),
    ("Where is the student film screening?", [("student film screening", EVT)]),
    ("When is the university graduation this year?", [("graduation", EVT)]),
    ("Is there a mental health awareness event on campus?", [("mental health awareness event", EVT)]),
    ("When is the freshers fair at the Student Union?", [("freshers fair", EVT), ("Student Union", LOC)]),
    ("Is there an international student welcome event?", [("international student welcome event", EVT)]),
    ("Where is the society recruitment fair?", [("society recruitment fair", EVT)]),
    ("When is the next guest lecture in computing?", [("guest lecture", EVT)]),
    ("Is there a sustainability fair this term?", [("sustainability fair", EVT)]),
    ("Where is the postgraduate research conference?", [("postgraduate research conference", EVT)]),
    ("When is the student union AGM?", [("student union AGM", EVT)]),
    ("Is the cafe in Ellison Building open on Saturday?", [("cafe", FAC), ("Ellison Building", LOC), ("Saturday", TIM)]),
    ("Where is the gym in the Sports Centre?", [("gym", FAC), ("Sports Centre", LOC)]),
    ("Is the library at City Campus open on Sunday?", [("library", FAC), ("City Campus", LOC), ("Sunday", TIM)]),
    ("When is the careers fair at the Student Union?", [("careers fair", EVT), ("Student Union", LOC)]),
    ("Is there a printer in Ellison Building?", [("printer", FAC), ("Ellison Building", LOC)]),
    ("What time does the cafe in the Student Union close on Friday?", [("cafe", FAC), ("Student Union", LOC), ("Friday", TIM)]),
    ("Is the gym open on weekends at Coach Lane Campus?", [("gym", FAC), ("weekends", TIM), ("Coach Lane Campus", LOC)]),
    ("Where is the nearest ATM to the Sutherland Building?", [("ATM", FAC), ("Sutherland Building", LOC)]),
    ("Is the graduation ceremony at City Campus this year?", [("graduation ceremony", EVT), ("City Campus", LOC)]),
    ("Is there a study room in the Lipman Building open late at night?", [("study room", FAC), ("Lipman Building", LOC), ("late at night", TIM)]),
    ("When is the open day at Coach Lane Campus?", [("open day", EVT), ("Coach Lane Campus", LOC)]),
    ("Is there a cafe near the Northumbria Library open on Sunday?", [("cafe", FAC), ("Northumbria Library", LOC), ("Sunday", TIM)]),
    ("Where is the computer lab in Ellison Building?", [("computer lab", FAC), ("Ellison Building", LOC)]),
    ("Is the freshers fair at the Student Union this week?", [("freshers fair", EVT), ("Student Union", LOC)]),
    ("What are the weekend hours for the gym at City Campus?", [("weekend", TIM), ("gym", FAC), ("City Campus", LOC)]),
    ("Is there a prayer room in the Sutherland Building?", [("prayer room", FAC), ("Sutherland Building", LOC)]),
    ("Where is the canteen in City Campus East?", [("canteen", FAC), ("City Campus East", LOC)]),
    ("Is the library in the Lipman Building open tomorrow?", [("library", FAC), ("Lipman Building", LOC), ("tomorrow", TIM)]),
    ("When is the next seminar in the Ellison Building?", [("seminar", EVT), ("Ellison Building", LOC)]),
    ("Is there a vending machine near City Campus East?", [("vending machine", FAC), ("City Campus East", LOC)]),
    ("Is the Northumbria Library open during freshers week?", [("Northumbria Library", LOC), ("freshers week", EVT)]),
    ("Is there parking near City Campus East?", [("parking", FAC), ("City Campus East", LOC)]),
    ("What is the closest cafe to the Northumbria Library?", [("cafe", FAC), ("Northumbria Library", LOC)]),
    ("Is the sports centre gym open on Bank Holiday?", [("Sports Centre", LOC), ("gym", FAC), ("Bank Holiday", TIM)]),
    ("Where is the nearest cafe to the Ellison Building open on Saturday morning?", [("cafe", FAC), ("Ellison Building", LOC), ("Saturday", TIM), ("morning", TIM)]),
    ("Is there a canteen in the Squires Building open on weekdays?", [("canteen", FAC), ("Squires Building", LOC), ("weekdays", TIM)]),
    ("When is the careers fair at City Campus this week?", [("careers fair", EVT), ("City Campus", LOC)]),
    ("Is there a study room in City Campus North open overnight?", [("study room", FAC), ("City Campus North", LOC), ("overnight", TIM)]),
    ("Where is the nearest photocopier to the Student Union?", [("photocopier", FAC), ("Student Union", LOC)]),
    ("Where can I print documents near the Student Union?", [("Student Union", LOC)]),
]

TRAIN_DATA = [make_example(text, ents) for text, ents in RAW]

BASE_CONFIG = """
[paths]
train = null
dev = null
vectors = null
init_tok2vec = null

[system]
gpu_allocator = null
seed = 0

[nlp]
lang = "en"
pipeline = ["tok2vec","ner"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = ${components.tok2vec.model.encode.width}
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,2500,2500,2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"
moves = null
update_with_oracle_cut_size = 100

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}
upstream = "*"

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${system.seed}
gpu_allocator = ${system.gpu_allocator}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 1e-8
learn_rate = 0.001

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vectors = ${paths.vectors}
init_tok2vec = ${paths.init_tok2vec}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""


def build_docbin(data, output_path):
    nlp = spacy.blank("en")
    db = DocBin()
    skipped = 0
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                skipped += 1
                continue
            ents.append(span)
        doc.ents = filter_spans(ents)
        db.add(doc)
    db.to_disk(output_path)
    print(f"Saved {len(data)} examples to {output_path} ({skipped} skipped)")


if __name__ == "__main__":
    print("=" * 60)
    print("Priya NER Training Pipeline")
    print("=" * 60)

    os.makedirs("data/ner", exist_ok=True)
    os.makedirs("models/ner", exist_ok=True)

    random.seed(42)
    random.shuffle(TRAIN_DATA)
    split = int(len(TRAIN_DATA) * 0.8)
    train_data = TRAIN_DATA[:split]
    dev_data = TRAIN_DATA[split:]

    print(f"\nDataset: {len(TRAIN_DATA)} examples | Train: {len(train_data)} | Dev: {len(dev_data)}")

    print("\nBuilding training data...")
    build_docbin(train_data, "data/ner/train.spacy")
    build_docbin(dev_data, "data/ner/dev.spacy")

    config_path = "data/ner/config.cfg"
    with open(config_path, "w") as f:
        f.write(BASE_CONFIG.strip())
    print(f"Config written to {config_path}")

    print("\nTraining NER model (1-2 minutes)...")
    result = subprocess.run([
        sys.executable, "-m", "spacy", "train",
        config_path,
        "--output", "models/ner",
        "--paths.train", "data/ner/train.spacy",
        "--paths.dev", "data/ner/dev.spacy",
    ])

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("SUCCESS - Model saved to models/ner/model-best")
        print("=" * 60)
    else:
        print("\nTraining failed. Check output above.")