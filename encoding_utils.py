import re
import json

# Read the formats-data.ts file
def parse_formats_data(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()

    # Extract the JSON-like structure
    data_match = re.search(r"export const FormatsData = ({.*});", file_contents, re.DOTALL)
    if not data_match:
        raise ValueError("Failed to extract FormatsData from the file.")

    formats_data_str = data_match.group(1)

    # Replace keys with valid JSON format
    formats_data_str = re.sub(r"([a-zA-Z0-9_]+):", r'"\1":', formats_data_str)

    # Remove trailing commas
    formats_data_str = re.sub(r",\s*}", "}", formats_data_str)  # Trailing commas in objects
    formats_data_str = re.sub(r",\s*]", "]", formats_data_str)  # Trailing commas in arrays

    # Convert to a Python dictionary
    formats_data = json.loads(formats_data_str)
    return formats_data

def get_filtered_pokemon_list(formats_data, excluded_tier="Uber"):
    return [
        name.capitalize()
        for name, details in formats_data.items()
        if details.get("tier") != excluded_tier
    ]

def get_pokemon_list():
    file_path = 'formats-data.ts'
    formats_data = parse_formats_data(file_path)
    pokemon_list = get_filtered_pokemon_list(formats_data)
    return pokemon_list

def get_move_list():
    moves_list = ["Absorb", "Acid", "Acid Armor", "Agility", "Amnesia", "Ancient Power", "Astonish", "Attract", "Barrage", "Barrier", "Baton Pass", "Beat Up", "Belly Drum", "Bide", "Bind", "Bite", "Blizzard", "Body Slam", "Bone Club", "Bonemerang", "Bounce", "Brick Break", "Bubble", "Bubble Beam", "Bulk Up", "Bullet Seed", "Calm Mind", "Camouflage", "Charge", "Clamp", "Comet Punch", "Confuse Ray", "Confusion", "Conversion", "Conversion 2", "Cosmic Power", "Counter", "Covet", "Crabhammer", "Cross Chop", "Crunch", "Crush Claw", "Curse", "Cut", "Defense Curl", "Destiny Bond", "Detect", "Dig", "Disable", "Dive", "Dizzy Punch", "Doom Desire", "Double-Edge", "Double Kick", "Double Team", "Doubleslap", "Dragon Breath", "Dragon Claw", "Dragon Dance", "Dragon Rage", "Drill Peck", "Dynamic Punch", "Earthquake", "Endeavor", "Endure", "Eruption", "Explosion", "Extrasensory", "Extremespeed", "Facade", "Fake Out", "Fake Tears", "False Swipe", "Feather Dance", "Fire Blast", "Fire Punch", "Fire Spin", "Fissure", "Flail", "Flame Wheel", "Flamethrower", "Flash", "Fly", "Focus Energy", "Focus Punch", "Follow Me", "Foresight", "Frustration", "Fury Attack", "Fury Cutter", "Fury Swipes", "Future Sight", "Giga Drain", "Glare", "Grasswhistle", "Growl", "Growth", "Guillotine", "Gust", "Hail", "Harden", "Haze", "Headbutt", "Heat Wave", "Helping Hand", "Hidden Power", "High Jump Kick", "Horn Attack", "Horn Drill", "Howl", "Hydro Cannon", "Hydro Pump", "Hyper Beam", "Hyper Fang", "Hypnosis", "Ice Beam", "Ice Ball", "Icicle Spear", "Icy Wind", "Imprison", "Iron Defense", "Iron Tail", "Jump Kick", "Karate Chop", "Kinesis", "Knock Off", "Leaf Blade", "Leer", "Lick", "Light Screen", "Lock-On", "Lovely Kiss", "Low Kick", "Lucky Chant", "Magical Leaf", "Magnet Rise", "Magnitude", "Mega Drain", "Mega Kick", "Mega Punch", "Megahorn", "Metal Claw", "Metal Sound", "Meteor Mash", "Mimic", "Mind Reader", "Minimize", "Miracle Eye", "Mirror Coat", "Mirror Move", "Mist", "Moonlight", "Morning Sun", "Mud Shot", "Mud-Slap", "Muddy Water", "Nature Power", "Needle Arm", "Nightmare", "Octazooka", "Odor Sleuth", "Outrage", "Overheat", "Pain Split", "Pay Day", "Perish Song", "Petal Dance", "Pin Missile", "Poison Fang", "Poison Gas", "Poison Jab", "Poison Sting", "Poison Tail", "Pound", "Powder Snow", "Present", "Protect", "Psybeam", "Psych Up", "Psychic", "Psycho Boost", "Pursuit", "Quick Attack", "Rage", "Rain Dance", "Rapid Spin", "Razor Leaf", "Razor Wind", "Recover", "Recycle", "Reflect", "Refresh", "Rest", "Return", "Revenge", "Reversal", "Roar", "Rock Blast", "Rock Polish", "Rock Slide", "Rock Smash", "Rock Tomb", "Role Play", "Rolling Kick", "Rollout", "Roost", "Sacred Fire", "Safeguard", "Sand Attack", "Sand Tomb", "Sandstorm", "Scary Face", "Scratch", "Screech", "Seismic Toss", "Self-Destruct", "Shadow Ball", "Shadow Claw", "Shadow Force", "Shadow Punch", "Sharpen", "Sheer Cold", "Shock Wave", "Signal Beam", "Silver Wind", "Sing", "Sketch", "Skill Swap", "Skull Bash", "Sky Attack", "Sky Uppercut", "Slam", "Slash", "Sleep Powder", "Sleep Talk", "Sludge", "Sludge Bomb", "SmellingSalt", "Smog", "Snatch", "Snore", "Soft-Boiled", "Solarbeam", "SonicBoom", "Spark", "Spider Web", "Spike Cannon", "Spikes", "Spit Up", "Spite", "Splash", "Spore", "Steel Wing", "Stomp", "Stone Edge", "Strength", "String Shot", "Struggle", "Stun Spore", "Submission", "Substitute", "Sucker Punch", "Sunny Day", "Super Fang", "Supersonic", "Surf", "Swagger", "Swallow", "Sweet Kiss", "Sweet Scent", "Swift", "Swords Dance", "Tackle", "Tail Glow", "Tail Whip", "Take Down", "Taunt", "Teleport", "Thief", "Thrash", "Thunder", "Thunder Wave", "Thunderbolt", "Thunderpunch", "Thundershock", "Tickle", "Toxic", "Transform", "Tri Attack", "Trick", "Trick Room", "Triple Kick", "Twineedle", "Twister", "Uproar", "Vacuum Wave", "Vicegrip", "Vine Whip", "Vital Throw", "Volt Tackle", "Water Gun", "Water Pulse", "Waterfall", "Weather Ball", "Whirlwind", "Will-O-Wisp", "Wing Attack", "Wish", "Withdraw", "Wonder Room", "Wrap", "Wring Out", "X-Scissor", "Yawn", "Zap Cannon", "Zen Headbutt"]
    return moves_list

def get_status_list():
    status_list = ["tox", "par", "brn", "frz", "slp", "psn"]
    return status_list