import pandas as pd
from tqdm.notebook import tqdm
from utilities import get_types, get_status_score, get_best_stab_multiplier, get_effect_score, construct_dictionary, reconstruct_team2
from staticFeatures import create_static_features
from dynamicFeatures import create_dynamic_features


def create_last_turn_features(features: dict, last_turn_data: dict, pokemon_dictionary: dict)-> dict:
    """
    Main function that extracts features from the state of the battle at the very last
    turn provided (e.g., turn 30).

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        last_turn_data (dict): The dictionary containing all the information (states and moves) about the last turn(the 30th)
        pokemon_dictionary (dict): The dictionary created before with all the information about pokemon.
        
    Returns:
        features (dict): The updated features dictionary with the status move counts features.
    """
    
    p1_state = last_turn_data.get('p1_pokemon_state', {})
    p2_state = last_turn_data.get('p2_pokemon_state', {})

    # Don't do anything if there are no information about last turn
    if not p1_state or not p2_state:
        return features

    ## Compute current HP difference
    p1_hp_pct = p1_state.get('hp_pct', 0)
    p2_hp_pct = p2_state.get('hp_pct', 0)
    features['active_hp_pct_diff'] = p1_hp_pct - p2_hp_pct

    ## Compute current boosts difference
    # Get current boosts information
    p1_boosts = p1_state.get('boosts', {})
    p2_boosts = p2_state.get('boosts', {})
    
    # Sum all 5 boost stages and compute the difference between the two players
    p1_boost_sum = sum(p1_boosts.values())
    p2_boost_sum = sum(p2_boosts.values())
    features['active_boost_sum_diff'] = p1_boost_sum - p2_boost_sum

    ## Compute effect score difference
    # Get current effect score
    p1_effects = p1_state.get('effects', ['noeffect'])
    p2_effects = p2_state.get('effects', ['noeffect'])
    
    p1_effect_score = get_effect_score(p1_effects)
    p2_effect_score = get_effect_score(p2_effects)
    
    # Compute the effect score difference: Positive score = P1 has an advantage 
    features['active_effect_advantage_score'] = p1_effect_score - p2_effect_score

    ## Compute base stats difference
    p1_name = p1_state.get('name')
    p2_name = p2_state.get('name')
    
    if p1_name and p2_name:
        p1_details = pokemon_dictionary[p1_name]
        p2_details = pokemon_dictionary[p2_name]
        
        # Calculate difference for the 5 stats
        features['active_base_hp_diff'] = p1_details.get('base_hp', 0) - p2_details.get('base_hp', 0)
        features['active_base_atk_diff'] = p1_details.get('base_atk', 0) - p2_details.get('base_atk', 0)
        features['active_base_def_diff'] = p1_details.get('base_def', 0) - p2_details.get('base_def', 0)
        features['active_base_special_diff'] = p1_details.get('base_spa', 0) - p2_details.get('base_spa', 0)
        features['active_base_spe_diff'] = p1_details.get('base_spe', 0) - p2_details.get('base_spe', 0)

        ## Compute Active Type Matchup Difference
        # Get types of current pokemon
        p1_active_types = get_types(p1_details)
        p2_active_types = get_types(p2_details)

        # How well P1's hit P2's active mon
        p1_best_mult = get_best_stab_multiplier(p1_active_types, p2_active_types)
        # How well P2's hit P1's active mon
        p2_best_mult = get_best_stab_multiplier(p2_active_types, p1_active_types)
        
        # Compute the "Active Threat Score"
        features['active_matchup_diff'] = p1_best_mult - p2_best_mult

    ## Compute the active status score difference
    p1_status = p1_state.get('status', 'nostatus')
    p2_status = p2_state.get('status', 'nostatus')
    features['active_status_advantage_score'] = get_status_score(p2_status) - get_status_score(p1_status)

    return features



def create_features(data: list[dict]) -> pd.DataFrame:
    """
    Applies the previous functions to all the battles in the dataset to create features for each battle.

    Parameter:
        data (list[dict]): List containing all the battles as dictionaries

    Returns:
        Pandas dataframe containing the features for all the battles in data
    """
    
    # First, construct the pokemon dictionary
    pokemon_dict = construct_dictionary(data)
    
    feature_list = []
    # Iterate through all the battles to create features for each one of them
    for battle in tqdm(data, desc="Extracting features"):
        features = {}

        battle_timeline = battle.get("battle_timeline", [])

        # Extract the information about P2's team
        p2_team = reconstruct_team2(battle_timeline, pokemon_dict)
        
        p1_team = battle.get('p1_team_details', [])

        # Create all the features applying previous functions
        features = create_static_features(features, p1_team, p2_team)
        features = create_last_turn_features(features, battle_timeline[-1], pokemon_dict)
        features = create_dynamic_features(features, battle_timeline, p1_team, p2_team, pokemon_dict)

        # We also need the ID and the target variable (if it exists)
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        # Append the features (dictionaries) to the list containing the features of each battle
        feature_list.append(features)

    # Return the pandas dataframe containing all the features
    return pd.DataFrame(feature_list).fillna(0)