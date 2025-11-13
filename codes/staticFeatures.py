import numpy as np
from collections import defaultdict
from utilities import get_types, TYPES, get_best_stab_multiplier, SUPER_EFFECTIVE


def create_static_features(features: dict, p1_team: list[dict], p2_team: list[dict])-> dict:
    """
    Acts as a master function to generate all static features for a single battle.
    
    It calls several helper functions to calculate team-level stats, 
    type advantages, and role compositions (they will be better explained in the next sections).

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.

    Returns:
        dict: The updated features dictionary containing all static features.
    """
    
    features = stat_features(features, p1_team, p2_team)
    features = type_counts(features, p1_team, p2_team)
    features = role_counts(features, p1_team, p2_team)
    features = team_matchup_scores(features, p1_team, p2_team)
    features = super_effective_threat_counts(features, p1_team, p2_team)

    return features



def get_team_stats(team: list[dict])-> dict:
    """
    Computes the mean and max for a given team.

    Parameter:
        team (list[dict]): The list of all the pokemon of a team and their characteristics

    Returns:
        stats_data (dict): A dictionary containing the mean and max for each stat of the team
    """
    stats_data = {}

    # Handle the case where there are no information about a team
    if not team or len(team) == 0:
        stat_keys = ['hp', 'spe', 'atk', 'def', 'special']
        for key in stat_keys:
            stats_data[f"mean_{key}"] = 0
            stats_data[f"max_{key}"] = 0
            stats_data[f"min_{key}"] = 0
        return stats_data

    # Get all the stats for each pokemon in the team
    stats = {
        'hp': [p.get('base_hp', 0) for p in team],
        'spe': [p.get('base_spe', 0) for p in team],
        'atk': [p.get('base_atk', 0) for p in team],
        'def': [p.get('base_def', 0) for p in team],
        'special': [p.get('base_spa', 0) for p in team]
    }
    
    stat_keys = ['hp', 'spe', 'atk', 'def', 'special']

    # Compute the mean and max for the team
    for key in stat_keys:
        stats_data[f"mean_{key}"] = np.mean(stats[key])
        stats_data[f"max_{key}"] = np.max(stats[key])
        
    return stats_data

def stat_features(features: dict, p1_team: list[dict], p2_team: list[dict])-> dict:
    """
    Computes the stat features of the two teams using the previously defined fucntion
    and the number of revealed pokemon for the player 2 team.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.

    Returns:
        dict: The updated features dictionary containing the stat features.
        
    """
    
    # Get full team stats of player 1
    p1_stats = get_team_stats(p1_team)
    for key, value in p1_stats.items():
        features[f"p1_team_{key}"] = value
        
    # Get revealed team stats of player 2
    p2_stats = get_team_stats(p2_team)
    for key, value in p2_stats.items():
        features[f"p2_team_{key}"] = value
        
    # This feature is essential because it tells the model how much information we have about player 2 team
    features["p2_revealed_count"] = len(p2_team)

    return features



def type_counts(features: dict, p1_team: list[dict], p2_team: list[dict])-> dict:
    """
    Calculates type-related features (psychic count, type diversity) 
    for both P1's full team and P2's revealed team. 
    This function applies an internal helper function to both teams.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.

    Returns:
        dict: The updated features dictionary containing the type features.
    """
    
    def get_team_type_counts(team: list[dict], player_prefix: str)-> None:
        """
        Internal helper function to process a single team.
        It calculates the counts and modifies the features dictionary without returning anything.
        
        Parameters:
            team (list[dict]): The list of Pokemon to process.
            player_prefix (str): The prefix to use for feature names ("p1" or "p2").
        """
        # defaultdict automatically assigns 0-count to not yet appeared types
        type_counts = defaultdict(int)

        # Iterate through the pokemon in the team and compute the number of pokemon for each type
        for pokemon in team:
            
            # Clean the types by making them lowercase and deleting notype
            types = [t.lower() for t in pokemon.get('types', []) if t.lower() != 'notype']
            for p_type in types:
                if p_type in TYPES:
                    type_counts[p_type] += 1

        # Store the Psychihc count feature
        features[f"{player_prefix}_psychic_count"] = type_counts["psychic"]
            
        # Count how many different types in the team and store the type diversity
        counter = 0
        for el in type_counts.values():
            if el:
                counter += 1
        features[f"{player_prefix}_type_diversity"] = counter

    # Calculate and store counts for P1's full team
    get_team_type_counts(p1_team, "p1")
    
    # Calculate and store counts for P2's revealed team
    get_team_type_counts(p2_team, "p2")

    return features



# Define thresholds for roles in the team
HIGH_SPEED = 100
HIGH_ATTACK = 100 
HIGH_SPECIAL = 100 

HIGH_EHP_PHYSICAL = 10000 
HIGH_EHP_SPECIAL = 10000 


def get_role_counts(team: list[dict]) -> dict:
    """
    Helper function to count the number of roles on a single team.

    Parameter:
        team (list[dict]): The list of Pokemon to process.

    Returns:
        roles (dict): A dictionary containing the count for each role.
    """
    # Define the dictionary with 0-count for each role
    roles = {
        'phys_sweeper': 0,
        'spec_sweeper': 0,
        'phys_wall': 0,
        'spec_wall': 0,
    }

    # Handle the case where there is no information about the team
    if not team:
        return roles

    # Iterate through the pokemon, get their stats to check if they have a role
    for p in team:
        # Get stats, default to 0 if missing
        hp = p.get('base_hp', 0)
        atk = p.get('base_atk', 0)
        defe = p.get('base_def', 0)
        spe = p.get('base_spe', 0)
        special = p.get('base_spa', 0) 

        ### Check if pokemon has one an important role by comparing its stats with the predefined thresholds

        # 1. Physical Sweeper (High Attack + High Speed)
        if atk >= HIGH_ATTACK and spe >= HIGH_SPEED:
            roles['phys_sweeper'] += 1
            
        # 2. Special Sweeper (High Special + High Speed)
        if special >= HIGH_SPECIAL and spe >= HIGH_SPEED:
            roles['spec_sweeper'] += 1
            
        # 3. Physical Wall (High Physical Effective HP)
        if (hp * defe) >= HIGH_EHP_PHYSICAL:
            roles['phys_wall'] += 1
            
        # 4. Special Wall (High Special Effective HP)
        if (hp * special) >= HIGH_EHP_SPECIAL:
            roles['spec_wall'] += 1
            
    return roles


def role_counts(features: dict, p1_team: list[dict], p2_team: list[dict])-> dict:
    """
    Calculates and adds role counts for P1's full team and P2's revealed team.
    This function modifies the dictionary containing all the features.
    
    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.

    Returns:
        dict: The updated features dictionary containing the role features.
    """
    
    # Get role counts for P1's full team
    p1_roles = get_role_counts(p1_team)
        
    features["p1_phys_sweeper"] = p1_roles['phys_sweeper']
    features["p1_spec_sweeper"] = p1_roles['spec_sweeper']
    features["p1_phys_wall"] = p1_roles['phys_wall']
    features["p1_spec_wall"] = p1_roles['spec_wall']
        
    # Get role counts for P2's revealed team
    p2_roles = get_role_counts(p2_team)
    
    features["p2_phys_sweeper"] = p2_roles['phys_sweeper']
    features["p2_spec_sweeper"] = p2_roles['spec_sweeper']
    features["p2_phys_wall"] = p2_roles['phys_wall']
    features["p2_spec_wall"] = p2_roles['spec_wall']

    return features

    

def team_matchup_scores(features: dict, p1_team: list[dict], p2_team_revealed: list[dict])-> dict:
    """
    Calculates the average "best STAB multiplier" for all pairwise
    matchups between P1's full team and P2's revealed team.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.

    Returns:
        dict: The updated features dictionary containing the matchup score features.
    """
    
    def calculate_team_offensive_score(attacking_team: list[dict], defending_team: list[dict])-> float:
        """
        Calculates the average best-case STAB multiplier for an
        attacking team against a defending team.

        It iterates through the attackers, compute their average best multiplier against all the defenders
        and averages all these results.

        Parameters:
            attacking_team (list[dict]): List of attacking team pokemon.
            defending_team (list[dict]): List of defending team pokemon.

        Returns:
            average best multiplier for attacking team against defending team
        """
        if not attacking_team or not defending_team:
            return 0.0
            
        total_score = 0.0

        # Iterates through the pokemon of the attacking team and computes their average effectiveness
        for attacker_mon in attacking_team:
            attacker_types = get_types(attacker_mon)
            
            # Find the average score for this one attacker against the whole enemy team
            attacker_avg_score = 0.0
            for defender_mon in defending_team:
                defender_types = get_types(defender_mon)
                attacker_avg_score += get_best_stab_multiplier(attacker_types, defender_types)
                
            total_score += (attacker_avg_score / len(defending_team))
            
        # Return the final average score for the whole team
        return total_score / len(attacking_team)

    # Calculate P1's offensive score vs. P2's revealed team
    p1_vs_p2_score = calculate_team_offensive_score(p1_team, p2_team_revealed)
    
    # Calculate P2's revealed team's offensive score vs. P1's full team
    p2_vs_p1_score = calculate_team_offensive_score(p2_team_revealed, p1_team)
    
    features["team_p1_vs_p2_revealed_offense_score"] = p1_vs_p2_score
    features["team_p2_revealed_vs_p1_offense_score"] = p2_vs_p1_score
    # Computes the difference between the two scores
    features["team_offense_score_diff"] = p1_vs_p2_score - p2_vs_p1_score

    return features        



def super_effective_threat_counts(features: dict, p1_team: list[dict], p2_team: list[dict])-> dict:
    """
    Counts how many Pokémon on each team have a STAB move that is
    super-effective against at least one member of the opposing team.

    Parameters:
        features (dict): The dictionary to which all new features will be added.
        p1_team (list[dict]): The full list of 6 Pokemon dictionaries for Player 1.
        p2_team (list[dict]): The list of revealed Pokemon dictionaries for Player 2.

    Returns:
        dict: The updated features dictionary containing the threat count features.
    """
    
    def calculate_threat_count(attacking_team: list[dict], defending_team: list[dict])-> int:
        """
        Counts the number of attackers that are super effective against
        at least one defender.

        Parameters:
            attacking_team (list[dict]): List of attacking team pokemon.
            defending_team (list[dict]): List of defending team pokemon.

        Returns:
            threat_count (int): The number of attackers that are super effective against at least one defender
        """
        if not attacking_team or not defending_team:
            return 0
            
        threat_count = 0
        for attacker_mon in attacking_team:
            attacker_types = get_types(attacker_mon)
            is_a_threat = False
            
            for defender_mon in defending_team:
                defender_types = get_types(defender_mon)
                
                # Check if this attacker's STAB hits this defender super-effectively
                best_mult = get_best_stab_multiplier(attacker_types, defender_types)
                
                if best_mult >= SUPER_EFFECTIVE:
                    is_a_threat = True
                    break  # Found a target, count this attacker and move to the next one
            
            if is_a_threat:
                threat_count += 1
                
        return threat_count

    # P1's full team vs. P2's revealed team
    p1_threat_count = calculate_threat_count(p1_team, p2_team)
    
    # P2's revealed team vs. P1's full team
    p2_threat_count = calculate_threat_count(p2_team, p1_team)
    
    features["p1_team_se_threat_count"] = p1_threat_count
    features["p2_team_se_threat_count"] = p2_threat_count
    features["se_threat_count_diff"] = p1_threat_count - p2_threat_count

    return features