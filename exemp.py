import re
import random
import pandas as pd

x = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18']
y = {
    'X1': (-508.12, 45.533),
    'X2': (-17.692, 52.652),
    'X3': (-1543.8, 8259.4),
    'X4': (-631.71, 4972),
    'X5': (-17.692, 52.652),
    'X6': (-2321800, 10236000),
    'X7': (-204.3, 8259.4),
    'X8': (-17.692, 53.689),
    'X9': (-771.65, 123.94),
    'X10': (-17.692, 47.597),
    'X11': (-771.65, 123.94),
    'X12': (-60.742, 179.9),
    'X13': (-500.75, 8.835),
    'X14': (-204.3, 8262.3),
    'X15': (-551.11, 293.15),
    'X16': (-667.73, 288770),
    'X17': (-765.8, 165.95),
    'X18': (-6.469, 53433),
}

classe = ["1.0", "0.0"]
def generate_random_rules(num_rules):
    rules = []
    for _ in range(num_rules):
        num_features = random.randint(2, 18)
        features = random.sample(x, num_features)
        operations = [random.choice(['=', '<', '>']) for _ in range(num_features)]
        values = [random.uniform(y[feature][0], y[feature][1]) for feature in features]
        conditions = ' and '.join([f'{feature} {operation} {value}' for feature, operation, value in zip(features, operations, values)])
        rule = f'If {conditions} than classe = {random.choice(classe)}'
        rules.append(rule)
         # Print the generated rule
        print(rule)
        print('\n')
        
    return rules
def validate_rules(rules, csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Initialize an empty list to store validated rules
    validated_rules = []

    # Iterate through each rule in the input list of rules
    for rule in rules:
        # Extract the conditions part of the rule (after 'If' and before 'than')
        conditions = rule.split('If ')[1].split(' than ')[0]

        # Split the conditions into a list of individual conditions
        conditions_list = conditions.split(' and ')

        # Extract features, operations, and values from each condition
        features = [c.split(' ')[0] for c in conditions_list]
        operations = [c.split(' ')[1] for c in conditions_list]
        values = [float(c.split(' ')[2]) for c in conditions_list]

        # Initialize a mask with True values for all rows in the DataFrame
        mask = pd.Series([True] * len(df))

        # Apply each condition to the mask to filter the rows
        for feature, operation, value in zip(features, operations, values):
            if operation == '<':
                mask &= df[feature] < value
            elif operation == '>':
                mask &= df[feature] > value
            elif operation == '=':
                mask &= df[feature] == value

        # Extract the class value from the rule using regular expressions
        class_str = re.search('(?<=classe = )[0-9.]+', rule).group(0)
        class_value = float(class_str)

        # Apply the class condition to the mask
        indices = mask & (df['classe'] == class_value)

        # Count the number of rows that satisfy the rule
        count = indices.sum()

        # If at least one row satisfies the rule, add the rule to the validated_rules list
        if count > 0:
            validated_rules.append(rule)

    # Return the list of validated rules
    return validated_rules
def create_solution(csv_path):
    solution=[]
    num_rules = random.randint(1000, 3000)  # Randomly determine the number of rules for each solution
    rules = generate_random_rules(num_rules)
    solution = validate_rules(rules, csv_path)
    return    solution
csv_path="modified_file.csv"
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        solution = create_solution(csv_path)
        population.append(solution)
    return population

def evaluate_solution(solution, csv_path):
    objective_values=[]
    min_validated_rules = float('inf')
    evaluated_rules = []
    base_df = pd.read_csv(csv_path)

    num_validated_rules = len(solution)
    if num_validated_rules < min_validated_rules:
        min_validated_rules = num_validated_rules

    solution_validated_rules_count = 0

    for rule in solution:
        rule_class_1_count = rule.count('classe = 1')

        if rule_class_1_count > 0:
            evaluated_rules.append(rule)

        solution_validated_rules_count += rule_class_1_count

    base_class_1_count = (base_df['classe'] == 1.0).sum()

    #n = len(solution)
    q_p = (solution_validated_rules_count / base_class_1_count if base_class_1_count != 0 else 0) / 1
    objective1_value = -min_validated_rules
    objective2_value = q_p
    objective_values=[objective1_value, objective2_value]
    return  objective_values
def evaluate_all_solutions(population, csv_path):
    objective_values = []

    for solution in population:
        _, objectives = evaluate_solution(solution, csv_path)
        objective_values.append(objectives)

    return objective_values

# Create the population and evaluate each solution
population_size=50
population = initialize_population(population_size)
solution = create_solution(csv_path)
objective_values=evaluate_all_solutions(population, csv_path)

def fast_nondominated_sort(population, objective_values):
    num_solutions = len(population)
    domination_count = [0] * num_solutions
    dominated_solutions = [[] for _ in range(num_solutions)]
    fronts = []  # List to store solutions for each front

    for i in range(num_solutions):
        fronts_i = []  # Initialize a list to store solutions for the current front
        for j in range(num_solutions):
            if i != j:
                # Check domination (i dominates j)
                if all(objective_values[i][k] <= objective_values[j][k] for k in range(2)):
                    dominated_solutions[i].append(j)
                # Check reverse domination (j dominates i)
                elif all(objective_values[j][k] <= objective_values[i][k] for k in range(2)):
                    domination_count[i] += 1

        if domination_count[i] == 0:
            fronts_i.append(i)
            # Attach the front number to the solution

        fronts.append(fronts_i)
    current_front = 0

    while len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
                    population[j].append(current_front + 1)  # Attach the front number to the solution

        current_front += 1
        fronts.append(next_front)

    return fronts

def crowding_distance_assignment(front, objective_values):
    num_solutions = len(front)
    crowding_distances = [0] * num_solutions

    # Sort the solutions in the front by each objective
    for i in range(2):  # Assuming there are two objectives
        sorted_indices = sorted(range(num_solutions), key=lambda x: objective_values[i][front[x]])

        # Check if the front is not empty
        if sorted_indices:
            # Set infinite crowding distances for boundary solutions
            crowding_distances[front[sorted_indices[0]]] = float('inf')
            crowding_distances[front[sorted_indices[-1]]] = float('inf')

            # Calculate crowding distance for other solutions
            for j in range(1, num_solutions - 1):
                crowding_distances[front[sorted_indices[j]]] += (objective_values[i][front[sorted_indices[j + 1]]] - objective_values[i][front[sorted_indices[j - 1]]])
    return crowding_distances



def binary_tournament_selection(fronts, crowding_distances, num_parents):
    selected_parents = []

    while len(selected_parents) < num_parents:
        # Randomly select two solutions from the fronts
        solution1 = random.choice(fronts)
        solution2 = random.choice(fronts)

        # Check non-dominance between the two solutions
        if dominates(solution1, solution2, crowding_distances):
            selected_parents.append(solution1)
        elif dominates(solution2, solution1, crowding_distances):
            selected_parents.append(solution2)
        else:
            # If they are not mutually dominant, compare crowding distances
            crowding1 = crowding_distances[fronts.index(solution1)]
            crowding2 = crowding_distances[fronts.index(solution2)]
            if crowding1 > crowding2:
                selected_parents.append(solution1)
            else:
                selected_parents.append(solution2)

    return selected_parents

def dominates(solution1, solution2, crowding_distances):
    # Check if solution1 dominates solution2 based on non-dominance and crowding distance
    dominates_solution1 = False
    dominates_solution2 = False

    for i in range(len(solution1)):
        if solution1[i] < solution2[i]:
            dominates_solution1 = True
        elif solution2[i] < solution1[i]:
            dominates_solution2 = True

    if dominates_solution1 and not dominates_solution2:
        return True
    if dominates_solution2 and not dominates_solution1:
        return False

    # If neither solution dominates the other, compare crowding distances
    crowding1 = crowding_distances[fronts.index(solution1)]
    crowding2 = crowding_distances[fronts.index(solution2)]

    return crowding1 > crowding2
fronts = fast_nondominated_sort(population, objective_values)
def one_point_crossover(parent1, parent2):
    # Choose a random crossover point
    crossover_point = random.randint(1, len(parent1) - 1)

    # Create two offspring by combining the parents at the crossover point
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

    return offspring1, offspring2
def mutation(solution, mutation_rate):
    mutated_solution = solution.copy()  # Make a copy of the solution

    # Apply mutation to some attributes based on the mutation rate
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            # Perform a random mutation operation on this attribute
            mutated_solution[i] = mutate_rule(mutated_solution[i])

    return mutated_solution

def mutate_rule(rule, mutation_rate, x, y, classe):
    # Check if mutation should occur based on the mutation rate
    if random.random() < mutation_rate:
        # Split the rule into its conditions and class parts
        conditions, class_part = rule.split(' than ')

        # Split the conditions into individual conditions
        conditions_list = conditions.split(' and ')

        # Choose a random condition to mutate
        condition_to_mutate = random.choice(conditions_list)

        # Split the condition into its feature, operation, and value
        feature, operation, value = condition_to_mutate.split()

        # Mutate the value of the condition within the valid range
        new_value = random.uniform(y[feature][0], y[feature][1])

        # Create the mutated condition
        mutated_condition = f'{feature} {operation} {new_value}'

        # Replace the old condition with the mutated condition
        conditions_list[conditions_list.index(condition_to_mutate)] = mutated_condition

        # Recreate the conditions part of the rule
        mutated_conditions = ' and '.join(conditions_list)

        # Recombine the mutated rule
        mutated_rule = f'If {mutated_conditions} than {class_part}'

        return mutated_rule
    else:
        # No mutation, return the original rule
        return rule




def nsga2(num_generations, population_size, mutation_rate, mutation_range):
    # Initialize the population
    population = initialize_population(population_size)

    for generation in range(num_generations):
        # Evaluate the objective values for the population
        objective_values = []
        for solution in population:
            _, objectives = evaluate_solution(solution, csv_path)
            objective_values.append(objectives)

        # Perform fast non-dominated sorting
        fronts = fast_nondominated_sort(population, objective_values)

        # Crowding distance assignment
        for front in fronts:
            crowding_distances = crowding_distance_assignment(front, objective_values)

        # Select parents using binary tournament selection
        num_parents = population_size  # Number of parents to select, equal to population size
        selected_parents = binary_tournament_selection(fronts, crowding_distances, num_parents)

        # Create offspring through crossover and mutation
        offspring = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]

            # Apply crossover to generate offspring
            offspring1, offspring2 = one_point_crossover(parent1, parent2)

            # Apply mutation to the offspring
            offspring1 = mutation(offspring1, mutation_rate, mutation_range)
            offspring2 = mutation(offspring2, mutation_rate, mutation_range)

            offspring.append(offspring1)
            offspring.append(offspring2)

        # Replace the old population with the new population (including parents and offspring)
        population = offspring

    return population

# Example usage:
num_generations = 20
population_size = 50
mutation_rate = 0.1
mutation_range = 0.1
csv_path = "modified_file.csv"

final_population = nsga2(num_generations, population_size, mutation_rate, mutation_range)
