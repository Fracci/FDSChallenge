from tqdm.notebook import tqdm


# The 15 types present in Generation 1
TYPES = {
    "normal", "fire", "water", "grass", "electric", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon"
}

# Gen 1 Type Chart to understand matchups among types
# Chart[Attacking_Type][Defending_Type] = Multiplier
TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0.0},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5},
    'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
    'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0.0, 'flying': 2.0, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2.0, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5},
    'ice': {'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 1.0}, # Note: Neutral to Dragon in RBY
    'fighting': {'normal': 2.0, 'ice': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2.0, 'ghost': 0.0},
    'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'bug': 2.0, 'rock': 0.5, 'ghost': 0.5}, # Note: Super-effective vs Bug
    'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0.0, 'bug': 0.5, 'rock': 2.0},
    'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5},
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'ghost': 0.0}, # Note: Immune to Ghost
    'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 2.0, 'flying': 0.5, 'psychic': 2.0, 'ghost': 0.5}, # Note: Super-effective vs Psychic & Poison
    'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0},
    'ghost': {'normal': 0.0, 'psychic': 0.0, 'ghost': 2.0}, # Note: Immune to Normal, No effect on Psychic
    'dragon': {'dragon': 2.0}
}


def construct_dictionary(data: list[dict])-> dict:
    """
    This fucntion builds a dictionary containing all the information (types, stats)
    about all the pokemon that appear in the battles.

    Parameter:
        data (list[dict]): List containing all the battles as dictionaries

    Returns:
        pokemon_dict (dict): Dictionary having as key the name of a pokemon and as value its information
    """
    pokemon_dict = {}

    # Iterate through all the battles to extract info about all the pokemon
    for battle in tqdm(data, desc = "creating dictionary"): 

        # Extract info about every pokemon of the first team and insert them in the dictionary with key = name of the pokemon
        for p in battle.get("p1_team_details", []): 
            pokemon_dict[p["name"]] = {
                "types": p["types"],
                "base_hp": p["base_hp"],
                "base_atk": p["base_atk"],
                "base_def": p["base_def"],
                "base_spa": p["base_spa"],
                "base_spd": p["base_spd"],
                "base_spe": p["base_spe"],
            }

        # Do the same thing with the only pokemon we know of the second team: the lead pokemon
        p2_lead = battle.get("p2_lead_details")
        name = p2_lead["name"]
        pokemon_dict[name] = {
            "types": p2_lead["types"],
            "base_hp": p2_lead["base_hp"],
            "base_atk": p2_lead["base_atk"],
            "base_def": p2_lead["base_def"],
            "base_spa": p2_lead["base_spa"],
            "base_spd": p2_lead["base_spd"],
            "base_spe": p2_lead["base_spe"],
        }

    return pokemon_dict


def reconstruct_team2(battle_timeline: list[dict], pokemon_dict: dict)-> list[dict]:
    """
    Reconstructs the team of player 2 by extracting the names of the pokemon used in the first 30 turns of a battle.

    Parameters:
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        pokemon_dictionary (dict): The dictionary created before with all the information about pokemon.

    Returns:
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.
    """
    p2_names = []

    # Iterate through the 30 turns and get the names of all the pokemon used by player 2 into a list (p2_names)
    for turn in battle_timeline:
        name = turn.get("p2_pokemon_state", {}).get("name")
        if name and name not in p2_names:
            p2_names.append(name)

    p2_team = []
    # For each name in p2_names associate the characteristics of the pokemon
    for name in p2_names:
        #if name in pokemon_dict.keys():
            d1 = {"name": name}
            d2 = pokemon_dict[name] # Pokemon type and stats are extracted from the dictionary created before
            d3 = d1 | d2

            p2_team.append(d3)
        
    return p2_team


def get_types(pokemon_details: dict)-> list[str]:
    """Safely extracts a list of lowercase types from a pokemon's data.

    Parameter:
        pokemon_details (dict): A dictionary containing the details of a pokemon.

    Returns:
        The list of types of that pokemon.
    """
    return [t.lower() for t in pokemon_details.get('types', []) if t.lower() != 'notype']


def get_status_score(status: str)-> int:
    """
    Assigns a penalty score to a status. A higher penalty is worse

    Parameter:
        status (str): String containing the status of the pokemon

    Returns:
        The penalty for the given status (int)
    """
    status_map = {
        'nostatus': 0, 'par': 2, 'slp': 3, 'frz': 3,
        'brn': 2, 'psn': 1, 'tox': 2
    }
    # This prevents errors in the access to the dictionary
    return status_map.get(status, 0)



# Helper function to get effect scores
def get_effect_score(effects_list: list[str])-> int:
    """
    Calculates a total score for a list of active effects.

    Parameter:
        effect_list (list[str]): List of effects of the current pokemon to analyze.

    Returns:
        total_score (int): Number containing the score of how good or bad are the current effects on the pokemon.
    """
    # Scores for common effects. Positive = good, Negative = bad
    EFFECT_SCORES = {
        'confusion': -2,
        'firespin': -3,
        'wrap': -3,
        'clamp': -3,
        'reflect': 2, 
        'lightscreen': 2,
        'leechseed': -1,
        'mist': 1,       
        'noeffect': 0
    }
    
    total_score = 0
    # Computes the sum of the effect scores
    for effect in effects_list:
        total_score += EFFECT_SCORES.get(effect.lower(), 0)
    return total_score



# Multipliers
SUPER_EFFECTIVE = 2.0
NOT_VERY_EFFECTIVE = 0.5
IMMUNE = 0.0
NORMAL = 1.0
    

def get_multiplier(attacking_type: str, defending_types: list[str])-> float:
    """
    Calculates the total multiplier for one attacking type vs. a list of defending types.

    Parameters:
        attacking_type (str): the type of the attacker move.
        defending_types (list[str]): the types of the defender pokemon.

    Returns:
        total_multiplier (float): the effectiveness of the attacker on the defender.
    """
    # Handle the case where there is no defending type
    if not defending_types:
        return NORMAL
        
    attack_chart = TYPE_CHART.get(attacking_type, {})
    total_multiplier = NORMAL

    # Computes the multiplier by multyplying the effects
    for def_type in defending_types:
        total_multiplier *= attack_chart.get(def_type, NORMAL)
        
    return total_multiplier
    

def get_best_stab_multiplier(attacker_types: list[str], defender_types: list[str])-> float:
    """
    Finds the best multiplier an attacker can get against a defender
    using only its own types.

    Parameters:
        attacking_type (str): the type of the attacker move.
        defending_types (list[str]): the types of the defender pokemon.

    Returns:
        best_multiplier (float): the best possible effectiveness of the attacker on the defender.
    """
    # Handle the case where there is no defending type
    if not attacker_types:
        return NORMAL
        
    best_multiplier = 0.0
    # Computes the best possible multiplier a pokemon can have against another
    for atk_type in attacker_types:
        multiplier = get_multiplier(atk_type, defender_types)
        best_multiplier = max(best_multiplier, multiplier)
        
    return best_multiplier