import numpy as np
from utilities import get_types, get_status_score


def create_dynamic_features(features: dict, battle_timeline: list[dict], p1_team: list[dict], p2_team: list[dict], pokemon_dictionary: dict)-> dict:
    """
    Acts as a master function to generate all dynamic features for a single battle.
    
    It calls several helper functions to calculate all the features we'll describe later.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.
        pokemon_dictionary (dict): The dictionary created before with all the information about pokemon.

    Returns:
        dict: The updated features dictionary containing all static features.
    """
    tuple1 = count_switches(features, battle_timeline)
    features = tuple1[0]
    p1_switch_list = tuple1[1]
    p2_switch_list = tuple1[2]
    
    features = get_skipped_turns(features, battle_timeline, p1_switch_list, p2_switch_list)

    tuple2 = create_pokemon_state_tracker(p1_team, p2_team, battle_timeline)
    pokemon_states = tuple2[0]
    p1_damage_list = tuple2[1]
    p2_damage_list = tuple2[2]
    p1_healing_list = tuple2[3]
    p2_healing_list = tuple2[4]
    
    features = get_hp_advantage(features, pokemon_states)
    features = count_fainted(features, pokemon_states)
    features = get_damage_features(features, p1_damage_list, p2_damage_list)
    features = get_recovery_features(features, p1_healing_list, p2_healing_list)

    features = get_status_move_counts(features, battle_timeline)
    features = get_status_move_counts(features, battle_timeline)
    features = get_total_status_features(features, pokemon_states)
    features = get_priority_move_counts(features, battle_timeline)
    features = get_move_power_features(features, battle_timeline, pokemon_dictionary)

    return features



def count_switches(features: dict, battle_timeline: list[dict])-> dict:
    """
    Calculates the total number of manual switches each player made,
    ignoring forced replacements from faints.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        
    Returns:
        features (dict): The updated features dictionary with the switches count features.
        p1_switch_list (list): 30-element list of P1's manual switches.
        p2_switch_list (list): 30-element list of P2's manual switches.
    """
    
    # Initialize 30-element lists with 0s (assume battle timeline is always of 30 turns)
    p1_switch_list = [0] * 30
    p2_switch_list = [0] * 30
    
    p1_switch_count = 0
    p2_switch_count = 0

    # Iterate through the turns to understand if a switch happened
    for i in range(1, 30):
        # Stop if we run out of turns in the timeline
        if i >= len(battle_timeline):
            break
            
        current_turn_data = battle_timeline[i]
        prev_turn_data = battle_timeline[i-1] # The state at the end of the previous turn

        # Get current names
        p1_name = current_turn_data.get('p1_pokemon_state', {}).get('name')
        p2_name = current_turn_data.get('p2_pokemon_state', {}).get('name')

        # Get previous names and HP
        prev_p1_state = prev_turn_data.get('p1_pokemon_state', {})
        prev_p2_state = prev_turn_data.get('p2_pokemon_state', {})
        
        prev_p1_name = prev_p1_state.get('name')
        prev_p1_hp = prev_p1_state.get('hp_pct', 0)
        
        prev_p2_name = prev_p2_state.get('name')
        prev_p2_hp = prev_p2_state.get('hp_pct', 0)

        # Check if there was a switch for player 1
        # Did the name change?
        if p1_name != prev_p1_name:
            # Was the previous pokemon ALIVE?
            if prev_p1_hp > 0:
                p1_switch_list[i] = 1
                p1_switch_count += 1

        # Check if there was a switch for player 2
        if p2_name != prev_p2_name:
            if prev_p2_hp > 0:
                p2_switch_list[i] = 1
                p2_switch_count += 1

    # Store the total counts as features
    features['p1_switch_count_total'] = p1_switch_count
    features['p2_switch_count_total'] = p2_switch_count
    features['switch_count_diff'] = p1_switch_count - p2_switch_count
    
    return features, p1_switch_list, p2_switch_list



def get_skipped_turns(features: dict, battle_timeline: list[dict], p1_switch_list: list[int], p2_switch_list: list[int])-> dict:
    """
    Calculates turns where a move was not made (null), excluding turns where a manual switch occurred.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        p1_switch_list (list): 30-element list of P1's manual switches.
        p2_switch_list (list): 30-element list of P2's manual switches.
        
    Returns:
        features (dict): The updated features dictionary with the skipped turns features.
    """
    
    # Lists to store "raw" null move turns
    p1_no_move_list = [0] * 30
    p2_no_move_list = [0] * 30
    
    # Final counts for features
    p1_skipped_count = 0
    p2_skipped_count = 0

    # Populate the "no move" lists by checking whether the move used was null
    # At the same time a switch must not occur in that turn to be sure that the move was null because of a status condition
    for i, turn_data in enumerate(battle_timeline[:30]):
        
        if turn_data.get('p1_move_details') is None and p1_switch_list[i] == 0:
            p1_skipped_count += 1
            
        if turn_data.get('p2_move_details') is None and p2_switch_list[i] == 0:
            p2_skipped_count += 1

    # Store the final counts as features
    features['p1_skipped_turns'] = p1_skipped_count
    features['p2_skipped_turns'] = p2_skipped_count
    features['skipped_turns_diff'] = p1_skipped_count - p2_skipped_count
    
    return features



# Define known Gen 1 recovery moves, that are only two
RBY_RECOVERY_MOVES = {'recover', 'rest'}

def create_pokemon_state_tracker(p1_team: list[dict], p2_team: list[dict], battle_timeline: list[dict])-> tuple:
    """
    Initializes a dictionary with all 12 Pokemon at 1.0 HP and 'nostatus'.
    Iterates through the timeline to update states.
    
    Also tracks:
    - Damage dealt by each player's damaging moves.
    - HP recovered by each player's healing moves.

    Parameters:
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.

    returns:
        A tuple containing:
            - pokemon_states (dict): The final state {'hp_pct', 'status'} of all 12 Pokemon.
            - p1_damage_list (list[float]): All damage instances dealt by P1.
            - p2_damage_list (list[float]): All damage instances dealt by P2.
            - p1_healing_list (list[float]): All healing instances for P1.
            - p2_healing_list (list[float]): All healing instances for P2.
    """
    
    pokemon_states = {}
    p1_damage_list = []
    p2_damage_list = []
    p1_healing_list = []
    p2_healing_list = []

    # Initialize P1's full team
    for pokemon in p1_team:
        p1_name = pokemon.get('name')
        if p1_name:
            key = f"p1_{p1_name}"
            pokemon_states[key] = {'hp_pct': 1.0, 'status': 'nostatus'}

    # Initialize P2's revealed team
    for pokemon in p2_team:
        p2_name = pokemon.get('name')
        if p2_name:
            key = f"p2_{p2_name}"
            pokemon_states[key] = {'hp_pct': 1.0, 'status': 'nostatus'}

    # Iterate through the timeline to compute everything said before
    for turn_data in battle_timeline[:30]:
        
        # Get states and moves from the current turn
        p1_state = turn_data.get('p1_pokemon_state', {})
        p1_name = p1_state.get('name')
        p1_key = f"p1_{p1_name}"
        
        p2_state = turn_data.get('p2_pokemon_state', {})
        p2_name = p2_state.get('name')
        p2_key = f"p2_{p2_name}"
        
        p1_move = turn_data.get('p1_move_details')
        p2_move = turn_data.get('p2_move_details')

        # Compute damage dealt by P1 move (only if it is a non-status move)
        if p1_move and p1_name and p2_name and p1_move.get('category') in ['PHYSICAL', 'SPECIAL']:
            # Compute the damage by doing the difference between the current hp and the previous hp of pokemon of P2
            old_hp = pokemon_states.get(p2_key, {'hp_pct': 1.0}).get('hp_pct', 1.0)
            new_hp = p2_state.get('hp_pct', old_hp)
            damage_dealt = old_hp - new_hp
            if damage_dealt > 0: 
                p1_damage_list.append(damage_dealt)

        # Compute damage dealt by P1 move (only if it is a non-status move)
        if p2_move and p1_name and p2_name and p2_move.get('category') in ['PHYSICAL', 'SPECIAL']:
            # Compute the damage by doing the difference between the current hp and the previous hp of pokemon of P1
            old_hp = pokemon_states.get(p1_key, {'hp_pct': 1.0}).get('hp_pct', 1.0)
            new_hp = p1_state.get('hp_pct', old_hp)
            damage_dealt = old_hp - new_hp
            if damage_dealt > 0: 
                p2_damage_list.append(damage_dealt)
        
        # Compute a possible healing only if the move used is a recovery move
        if p1_move and p1_name and p1_move.get('name', '').lower() in RBY_RECOVERY_MOVES:
            # Compute the healing by doing the difference between the previous and the current hp of pokemon of P1
            old_hp = pokemon_states.get(p1_key, {'hp_pct': 0.0}).get('hp_pct', 0.0)
            new_hp = p1_state.get('hp_pct', old_hp)
            hp_healed = new_hp - old_hp
            if hp_healed > 0:
                p1_healing_list.append(hp_healed)

        # Compute a possible healing only if the move used is a recovery move
        if p2_move and p2_name and p2_move.get('name', '').lower() in RBY_RECOVERY_MOVES:
            # Compute the healing by doing the difference between the previous and the current hp of pokemon of P2
            old_hp = pokemon_states.get(p2_key, {'hp_pct': 0.0}).get('hp_pct', 0.0)
            new_hp = p2_state.get('hp_pct', old_hp)
            hp_healed = new_hp - old_hp
            if hp_healed > 0:
                p2_healing_list.append(hp_healed)

        # Update P1 State (hp and status)
        if p1_name:
            pokemon_states[p1_key] = {
                'hp_pct': p1_state.get('hp_pct', 1.0),
                'status': p1_state.get('status', 'nostatus')
            }

        # Update P2 State (hp and status)
        if p2_name:
            pokemon_states[p2_key] = {
                'hp_pct': p2_state.get('hp_pct', 1.0),
                'status': p2_state.get('status', 'nostatus')
            }
            
    return pokemon_states, p1_damage_list, p2_damage_list, p1_healing_list, p2_healing_list



def count_fainted(features: dict, pokemon_states: dict)-> dict:
    """
    Counts the number of fainted Pokémon for each team using the
    final pokemon_states dictionary that contains the status of the pokemon.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        pokemon_states (dict): The final state {'hp_pct', 'status'} of all 12 Pokemon.

    Returns:
        features (dict): The updated features dictionary with the faint counts features.
    """
    
    p1_fainted_count = 0
    p2_fainted_count = 0

    # Iterate through the final state of all known Pokemon
    for key, state in pokemon_states.items():
        
        # A Pokemon is fainted if its HP is 0 or less
        # We use .get() with a default of 1.0 just in case hp_pct is missing
        if state.get('hp_pct', 1.0) <= 0:
            if key.startswith('p1_'):
                p1_fainted_count += 1
            elif key.startswith('p2_'):
                p2_fainted_count += 1

    # Store the features
    features['p1_fainted_count'] = p1_fainted_count
    features['p2_fainted_count'] = p2_fainted_count
    
    # A positive number here means P1 is winning (P2 has lost more Pokemon)
    features['fainted_count_diff'] = p2_fainted_count - p1_fainted_count
    
    return features



def get_hp_advantage(features: dict, pokemon_states: dict)-> dict:
    """
    Calculates the total HP advantage by comparing P1's full 6-Pokemon
    HP sum against P2's 6-Pokemon HP sum.
    
    This function imputes 1.0 HP for any of P2's Pokemon that
    have not yet been revealed in the first 30 turns.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        pokemon_states (dict): The final state {'hp_pct', 'status'} of all 12 Pokemon.

    Returns:
        features (dict): The updated features dictionary with the hp advantage features.
    """
    
    p1_total_hp_sum = 0.0
    p2_revealed_hp_sum = 0.0
    p2_revealed_count = 0

    # Sum the HP for both teams based on what's in the state tracker
    for key, state in pokemon_states.items():
        if key.startswith('p1_'):
            # P1's tracker is initialized with all 6, so this sum is correct
            p1_total_hp_sum += state.get('hp_pct', 0)
        elif key.startswith('p2_'):
            # This sums only the P2 Pokemon we've seen
            p2_revealed_hp_sum += state.get('hp_pct', 0)
            p2_revealed_count += 1

    # The unseen Pokemon of P2 all have 1.0 HP, so we have to take this into account
    p2_unseen_count = 6 - p2_revealed_count
    p2_unseen_hp = p2_unseen_count * 1.0
    
    p2_total_hp_sum = p2_revealed_hp_sum + p2_unseen_hp

    # Store the final, fair features
    
    # This is now a direct 6-vs-6 HP sum comparison
    features['total_hp_sum_diff'] = p1_total_hp_sum - p2_total_hp_sum

    # This is the 6-vs-6 HP ratio
    total_hp_pool = p1_total_hp_sum + p2_total_hp_sum
    if total_hp_pool > 0:
        features['hp_advantage_ratio'] = p1_total_hp_sum / total_hp_pool
    else:
        features['hp_advantage_ratio'] = 0.5
        
    return features



def get_damage_features(features: dict, p1_damage_list: list[float], p2_damage_list: list[float])-> dict:
    """
    Calculates the average damage dealt per damaging move used.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_damage_list (list[float]): All damage instances dealt by P1.
        p2_damage_list (list[float]): All damage instances dealt by P2.
        
    Returns:
        features (dict): The updated features dictionary with the damage features.
    """
    
    # Compute P1 average and max damage from the list
    if len(p1_damage_list) > 0:
        p1_avg_damage = np.mean(p1_damage_list)
        p1_max_damage = np.max(p1_damage_list)
    else:
        # Avoid division by zero if no damaging moves were used
        p1_avg_damage = 0.0
        p1_max_damage = 0.0
        
    # Compute P2 average and max damage from the list
    if len(p2_damage_list) > 0:
        p2_avg_damage = np.mean(p2_damage_list)
        p2_max_damage = np.max(p2_damage_list)
    else:
        p2_avg_damage = 0.0
        p2_max_damage = 0.0

    # Store the features
    features['p1_avg_damage'] = p1_avg_damage
    features['p2_avg_damage'] = p2_avg_damage
    features['p1_max_damage'] = p1_max_damage
    features['p2_max_damage'] = p2_max_damage
    features['avg_damage_diff'] = p1_avg_damage - p2_avg_damage
    features['max_damage_diff'] = p1_max_damage - p2_max_damage
    
    return features



def get_recovery_features(features: dict, p1_healing_list: list[float], p2_healing_list: list[float])-> dict:
    """
    Calculates features related to recovery moves, that are
    count and total HP healed.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_healing_list (list[float]): All healing instances for P1.
        p2_healing_list (list[float]): All healing instances for P2.
        
    Returns:
        features (dict): The updated features dictionary with the recovery features.
    """
    
    # P1 Recovery Stats: count of moves used, total hp healed and average healing
    p1_recovery_count = len(p1_healing_list)
    p1_total_hp_healed = np.sum(p1_healing_list)
    
    if p1_recovery_count > 0:
        p1_avg_hp_healed = p1_total_hp_healed / p1_recovery_count
    else:
        p1_avg_hp_healed = 0.0

    # P2 Recovery Stats: count of moves used and total hp healed
    p2_recovery_count = len(p2_healing_list)
    p2_total_hp_healed = np.sum(p2_healing_list)

    # Store features
    
    # Raw counts
    features['p1_recovery_count'] = p1_recovery_count
    features['p2_recovery_count'] = p2_recovery_count
    
    # Total HP recovered
    features['p1_total_hp_healed'] = p1_total_hp_healed
    features['p2_total_hp_healed'] = p2_total_hp_healed
    
    # Differences
    features['recovery_count_diff'] = p1_recovery_count - p2_recovery_count
    features['total_hp_healed_diff'] = p1_total_hp_healed - p2_total_hp_healed
    
    return features



def get_status_move_counts(features: dict, battle_timeline: list[dict])-> dict:
    """
    Calculates the total number of status moves used by each player
    during the first 30 turns.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        
    Returns:
        features (dict): The updated features dictionary with the status move counts features.
    """
    
    p1_status_move_count = 0
    p2_status_move_count = 0
    
    # Iterate through the timeline up to turn 30 (the last one we have information about)
    for turn_data in battle_timeline[:30]:
        
        # Check if P1 move is a status move and update the counter
        p1_move = turn_data.get('p1_move_details')
        if p1_move:
            p1_name = p1_move.get('name')
            if p1_move and p1_move.get('category') == 'STATUS' and p1_name != 'rest' and p1_name != 'recover':
                p1_status_move_count += 1

        # Check if P2 move is a status move and update the counter
        p2_move = turn_data.get('p2_move_details')
        if p2_move:
            p2_name = p2_move.get('name')
            if p2_move and p2_move.get('category') == 'STATUS' and p2_name != 'rest' and p2_name != 'recover':
                p2_status_move_count += 1

    # Store the features
    features['p1_status_move_count'] = p1_status_move_count
    features['p2_status_move_count'] = p2_status_move_count
    features['status_move_count_diff'] = p1_status_move_count - p2_status_move_count
    
    return features



def get_total_status_features(features: dict, pokemon_states: list[dict])-> dict:
    """
    Calculates total status counts and a total penalty score
    from the final pokemon_states dictionary.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        pokemon_states (dict): The final state {'hp_pct', 'status'} of all 12 Pokemon.

    Returns:
        features (dict): The updated features dictionary with the status score features.
    """
    
    # Initialize counts for major statuses
    p1_status_counts = {'par': 0, 'slp': 0, 'frz': 0}
    p2_status_counts = {'par': 0, 'slp': 0, 'frz': 0}
    
    # Initialize total penalty scores
    p1_total_penalty = 0
    p2_total_penalty = 0

    # Iterate through the final state of all known Pokemon and get their score
    for key, state in pokemon_states.items():
        status = state.get('status', 'nostatus')
        
        # Skip if 'nostatus' to save computation
        if status == 'nostatus':
            continue
            
        score = get_status_score(status)

        # Sum the scores for P1
        if key.startswith('p1_'):
            p1_total_penalty += score
            if status in p1_status_counts:
                p1_status_counts[status] += 1

        # Sum the scores for P2
        elif key.startswith('p2_'):
            p2_total_penalty += score
            if status in p2_status_counts:
                p2_status_counts[status] += 1
    
    # Store individual status counts
    for status, count in p1_status_counts.items():
        features[f'p1_total_{status}_count'] = count
        
    for status, count in p2_status_counts.items():
        features[f'p2_total_{status}_count'] = count
    
    # Store total penalty scores
    features['p1_total_status_penalty'] = p1_total_penalty
    features['p2_total_status_penalty'] = p2_total_penalty
    
    # Positive score means P1 has an advantage (P2 is more afflicted)
    features['total_status_advantage_score'] = p2_total_penalty - p1_total_penalty
    
    return features



def get_priority_move_counts(features: dict, battle_timeline: list[dict])-> dict:
    """
    Calculates the total number of priority moves (priority > 0)
    used by each player during the first 30 turns.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        
    Returns:
        features (dict): The updated features dictionary with the priority move counts features.
    """
    
    p1_priority_move_count = 0
    p2_priority_move_count = 0
    
    # Iterate through the timeline up to turn 30 (the last turn available for information)
    for turn_data in battle_timeline[:30]:
        
        # Check if P1 move has priority > 1
        p1_move = turn_data.get('p1_move_details')
        if p1_move and p1_move.get('priority', 0) > 0:
            p1_priority_move_count += 1

        # Check if P2 move has priority > 1
        p2_move = turn_data.get('p2_move_details')
        if p2_move and p2_move.get('priority', 0) > 0:
            p2_priority_move_count += 1

    # Store the features
    features['p1_priority_move_count'] = p1_priority_move_count
    features['p2_priority_move_count'] = p2_priority_move_count
    features['priority_move_count_diff'] = p1_priority_move_count - p2_priority_move_count
    
    return features



def get_move_power_features(features: dict, battle_timeline: list[dict], pokemon_dictionary: dict)-> dict:
    """
    Calculates the average and max power of all damaging moves (Physical/Special)
    used by each player, applying the 1.5x STAB bonus.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        battle_timeline (list[dict]): The list of the 30 revealed turns of the battle.
        
    Returns:
        features (dict): The updated features dictionary with the status move counts features.
    """
    
    p1_power_list = []
    p2_power_list = []

    # Iterate through the timeline up to turn 30 to get the power of each move used
    for turn_data in battle_timeline[:30]:

        # Extract move and state for each turn
        p1_move = turn_data.get('p1_move_details')
        p1_state = turn_data.get('p1_pokemon_state', {})
        
        p2_move = turn_data.get('p2_move_details')
        p2_state = turn_data.get('p2_pokemon_state', {})

        # Calculate P1's effective power
        if p1_move and p1_move.get('category') in ['PHYSICAL', 'SPECIAL']:
            # Extract the base power of the move
            base_power = p1_move.get('base_power', 0)
            
            if base_power > 0: # Only count moves with power (non-status moves)
                move_type = p1_move.get('type', '').lower()
                attacker_name = p1_state.get('name')
                
                if attacker_name:
                    # Get the attacker types
                    attacker_details = pokemon_dictionary.get(attacker_name, {})
                    attacker_types = get_types(attacker_details)
                    
                    effective_power = base_power
                    # Apply STAB if move type is one of the attacker's types
                    if move_type in attacker_types:
                        effective_power *= 1.5
                        
                    p1_power_list.append(effective_power)

        # Calculate P2's effective power
        if p2_move and p2_move.get('category') in ['PHYSICAL', 'SPECIAL']:
            base_power = p2_move.get('base_power', 0)
            if base_power > 0:
                move_type = p2_move.get('type', '').lower()
                attacker_name = p2_state.get('name')
                
                if attacker_name:
                    attacker_details = pokemon_dictionary.get(attacker_name, {})
                    attacker_types = get_types(attacker_details)
                    
                    effective_power = base_power
                    if move_type in attacker_types:
                        effective_power *= 1.5
                        
                    p2_power_list.append(effective_power)

    # Compute average power
    p1_avg_power = np.mean(p1_power_list) if len(p1_power_list) > 0 else 0.0
    p2_avg_power = np.mean(p2_power_list) if len(p2_power_list) > 0 else 0.0

    # Compute max power
    p1_max_power = np.max(p1_power_list) if len(p1_power_list) > 0 else 0.0
    p2_max_power = np.max(p2_power_list) if len(p2_power_list) > 0 else 0.0

    # Store the features
    features['p1_avg_move_power'] = p1_avg_power
    features['p2_avg_move_power'] = p2_avg_power
    features['avg_move_power_diff'] = p1_avg_power - p2_avg_power
    
    features['p1_max_move_power'] = p1_max_power
    features['p2_max_move_power'] = p2_max_power
    features['max_move_power_diff'] = p1_max_power - p2_max_power
    
    return features