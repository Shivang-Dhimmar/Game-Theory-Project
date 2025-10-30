import networkx as nx
from z3 import Solver, Real, Sum, sat, Implies
from collections import defaultdict
import random

random.seed(42)


def solve_traffic_equilibrium(nodes, edges, commodities):
    """
    Tries to apply https://en.wikipedia.org/wiki/John_Glen_Wardrop -> Wardrop equilibria -> Wardrop's First Principle
    (substitute cost instead of time) and finds a set of ticket prices and passenger flows that satisfy capacity and
    equilibrium constraints.

    Args:
        nodes (list): A list of node identifiers (e.g., ['A', 'B']).
        edges (list): A list of edge tuples (u, v, attr), where attr is a dict

                      containing 'k' (congestion factor), 'capacity', and
                      'price' (a number for a fixed price, or None for a var price to be calced).

        commodities (list): A list of commodity tuples (source, dest, demand).

    Returns:
        dict: A dictionary containing the solved prices, edge flows, path flows,
              and equilibrium costs. None if it cannot be found
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for u, v, attr in edges:
        G.add_edge(u, v, **attr)

    solver = Solver()

    # create a variable for each edge with an undefined price
    price_vars = {}
    for u, v, data in G.edges(data=True):
        if data.get("price") is None:
            p_var = Real(f"p_{u}_{v}")
            solver.add(p_var >= 0)  # negative prices too overpowered for this world
            price_vars[(u, v)] = p_var

    # create a variable for the flow on every possible path for each commodity
    path_flow_vars = defaultdict(list)
    commodity_paths = {}
    for i, (s, d, w) in enumerate(commodities):
        paths = list(nx.all_simple_paths(G, source=s, target=d))
        if not paths:
            print(f"no path found for commodity {i} from {s} to {d}. Ignore.")
            continue
        commodity_paths[i] = paths

        for j, path in enumerate(paths):
            f_var = Real(f"f_comm{i}_path{j}")
            solver.add(f_var >= 0)
            path_flow_vars[i].append(f_var)

        # add flow conservation constraint for the commodity
        solver.add(Sum(path_flow_vars[i]) == w)

    # total flow on each edge
    edge_flow_exprs = {}
    for u, v in G.edges():
        flow_on_edge = []
        for i, paths in commodity_paths.items():
            for j, path in enumerate(paths):
                # path = a list of nodes. check if edge (u,v) is in the path
                path_edges = list(zip(path[:-1], path[1:]))
                if (u, v) in path_edges:
                    flow_on_edge.append(path_flow_vars[i][j])

        edge_flow_exprs[(u, v)] = Sum(flow_on_edge) if flow_on_edge else Real(0)

    # dd capacity constraints
    for u, v, data in G.edges(data=True):
        if "capacity" in data and data["capacity"] != float("inf"):
            solver.add(edge_flow_exprs[(u, v)] <= data["capacity"])

    # add equilibrium constraints (indifference)
    equilibrium_costs = {}
    for i, (s, d, w) in enumerate(commodities):
        if i in commodity_paths and len(commodity_paths[i]) <= 1:
            print(f"{i} has only 1 route.")
        if i not in commodity_paths or len(commodity_paths[i]) <= 1:
            continue

        paths = commodity_paths[i]
        path_cost_exprs = []

        # substitute cost expression for every path available to the commodity
        for path in paths:
            path_edges = list(zip(path[:-1], path[1:]))
            cost_components = []
            for u, v in path_edges:
                edge_data = G.get_edge_data(u, v)
                k = edge_data["k"]
                p = price_vars.get((u, v), edge_data.get("price", 0))
                x = edge_flow_exprs[(u, v)]
                cost_components.append(k * x + p)
            path_cost_exprs.append(Sum(cost_components))

        # wardrop equilibrium, if a path is used (flow > 0), its cost must be equal to the minimum cost for that commodity.
        min_cost_var = Real(f"min_cost_comm{i}")
        for j, path in enumerate(paths):
            flow_var = path_flow_vars[i][j]
            cost_expr = path_cost_exprs[j]
            # if flow is positive, cost must equal the minimum.
            solver.add(Implies(flow_var > 0, cost_expr == min_cost_var))
            # cost of any path must be greater than or equal to the minimum.
            solver.add(cost_expr >= min_cost_var)

        equilibrium_costs[f"{s}->{d}"] = min_cost_var

    if solver.check() == sat:
        model = solver.model()
        solution = {
            "prices": {},
            "edge_flows": {},
            "path_flows": {},
            "equilibrium_costs": {},
        }

        for (u, v), p_var in price_vars.items():
            solution["prices"][(u, v)] = model.evaluate(
                p_var, model_completion=True
            ).as_decimal(3)

        for (u, v), x_expr in edge_flow_exprs.items():
            solution["edge_flows"][(u, v)] = model.evaluate(
                x_expr, model_completion=True
            ).as_decimal(3)

        for i, f_vars in path_flow_vars.items():
            s, d, _ = commodities[i]
            paths = commodity_paths[i]
            comm_key = f"({s}->{d})"
            solution["path_flows"][comm_key] = {}
            for j, f_var in enumerate(f_vars):
                flow_val = model.evaluate(f_var, model_completion=True).as_decimal(3)
                path_str = "->".join(paths[j])
                solution["path_flows"][comm_key][path_str] = flow_val

        for route, cost_expr in equilibrium_costs.items():
            solution["equilibrium_costs"][route] = model.evaluate(
                cost_expr, model_completion=True
            ).as_decimal(3)

        return solution
    else:
        return None


multigraph_edges = [
    # Line 1 - Blue Line (Dakshineshwar -> Kavi Subhash)
    ("Dakshineshwar", "Baranagar", "blue"),
    ("Baranagar", "Noapara", "blue"),
    ("Noapara", "Dum Dum", "blue"),
    ("Dum Dum", "Esplanade", "blue"),
    ("Esplanade", "Park Street", "blue"),
    ("Park Street", "Rabinder Sarobar", "blue"),
    ("Rabinder Sarobar", "Kavi Subhash", "blue"),
    # Line 2 - Green Line (Howrah Maidan -> Salt Lake)
    ("Howrah Maidan", "Sealdah", "green"),
    ("Sealdah", "Esplanade", "green"),
    (
        "Esplanade",
        "Salt Lake",
        "green",
    ),  # "Salt Lake" node assumed to be an interchange, e.g., Sec V
    # Line 3 - Purple Line (Joka -> Esplanade)
    ("Majherhat", "Diamond Park", "purple"),
    ("Diamond Park", "Park Street", "purple"),
    (
        "Park Street",
        "Esplanade",
        "purple",
    ),  # Note: Park St-Esplanade is a multigraph section (Blue & Purple)
    # Line 4 - Yellow Line (Noapara -> Barasat)
    ("Noapara", "Barasat", "yellow"),
    # Line 5 - Pink Line (Baranagar -> Barrackpore)
    ("Baranagar", "Mangal pandey", "pink"),  # Mangal Pandey is on this line
    # Line 6 - Orange Line (Kavi Subhash -> Jai Hind/Airport)
    ("Kavi Subhash", "Balygunj", "orange"),
    ("Balygunj", "Salt Lake", "orange"),
    ("Salt Lake", "Jai Hind", "orange"),
]

traffic = [
    # High Traffic (Hub-to-Hub / Hub-to-Rail)
    ("Howrah Maidan", "Sealdah", 2000),
    ("Dum Dum", "Esplanade", 1950),
    ("Sealdah", "Esplanade", 1900),
    ("Dum Dum", "Park Street", 1850),
    ("Howrah Maidan", "Esplanade", 1800),
    ("Kavi Subhash", "Esplanade", 1750),
    ("Kavi Subhash", "Park Street", 1700),
    ("Salt Lake", "Sealdah", 1650),
    ("Salt Lake", "Esplanade", 1600),
    ("Jai Hind", "Howrah Maidan", 1550),
    ("Jai Hind", "Esplanade", 1500),
    # Medium Traffic (Suburban-to-Hub / Airport-to-Suburban)
    ("Barasat", "Dum Dum", 1300),
    ("Barasat", "Esplanade", 1250),
    ("Dakshineshwar", "Esplanade", 1200),
    ("Rabinder Sarobar", "Howrah Maidan", 1150),
    ("Baranagar", "Park Street", 1100),
    ("Majherhat", "Esplanade", 1000),
    ("Majherhat", "Park Street", 950),
    ("Mangal pandey", "Esplanade", 900),
    ("Balygunj", "Park Street", 850),
    ("Diamond Park", "Park Street", 800),
    ("Kavi Subhash", "Balygunj", 750),
    ("Jai Hind", "Salt Lake", 700),
    # Low Traffic (Cross-Suburban / Multi-Transfer)
    ("Dakshineshwar", "Howrah Maidan", 650),
    ("Barasat", "Kavi Subhash", 600),
    ("Majherhat", "Sealdah", 550),
    ("Diamond Park", "Dum Dum", 500),
    ("Mangal pandey", "Jai Hind", 350),
    ("Mangal pandey", "Baranagar", 400),
    ("Dakshineshwar", "Majherhat", 300),
    ("Barasat", "Majherhat", 250),
    ("Rabinder Sarobar", "Balygunj", 200),
    ("Mangal pandey", "Diamond Park", 150),
]

def solve_for_one_network():
    nodes = []
    useful_edges = []
    for i, j, k in multigraph_edges:
        attr = {
            "k": 1.0,
            "capacity": 1e5,
            "price": None,
        }
        useful_edges.append((i, j, attr))
        useful_edges.append((j, i, attr))
        if i not in nodes:
            nodes.append(i)
        if j not in nodes:
            nodes.append(j)

    low = 0
    high = 1000000
    minimal_capacity = high

    while low <= high:
        mid = (low + high) // 2
        nodes = []
        useful_edges = []
        for i, j, k in multigraph_edges:
            attr = {
                "k": 1.0,
                "capacity": mid,
                "price": None,
            }
            useful_edges.append((i, j, attr))
            useful_edges.append((j, i, attr))
            if i not in nodes:
                nodes.append(i)
            if j not in nodes:
                nodes.append(j)

        solution = solve_traffic_equilibrium(nodes, useful_edges, traffic)
        if solution:
            minimal_capacity = mid
            high = mid - 1
        else:
            low = mid + 1

    print(f"\nMinimal capacity that keeps the model satisfiable: {minimal_capacity}")
    input()

    # run with the minimal capacity
    nodes = []
    useful_edges = []
    for i, j, k in multigraph_edges:
        attr = {
            "k": 1.0,
            "capacity": minimal_capacity,
            "price": None,
        }
        useful_edges.append((i, j, attr))
        useful_edges.append((j, i, attr))
        if i not in nodes:
            nodes.append(i)
        if j not in nodes:
            nodes.append(j)

    solution = solve_traffic_equilibrium(nodes, useful_edges, traffic)

    if solution:
        print("Success - Found a satisfying assignment for equilibrium:")

        print("\n\ncalculated metro ticket prices:")
        if solution["prices"]:
            for (u, v), price in solution["prices"].items():
                print(f"  - Price for {u}->{v}: {price}")
        else:
            print("  - No variable prices to solve for.")

        print("\n\nResulting edge flows at equilibrium:")

        # re-create graph to access capacity info
        G_sol = nx.DiGraph()
        for u, v, attr in useful_edges:
            G_sol.add_edge(u, v, **attr)
        for (u, v), flow in sorted(solution["edge_flows"].items()):
            capacity = G_sol.get_edge_data(u, v).get("capacity", "inf")
            status = "works" if float(flow) <= capacity else "RIP safety"
            print(f"  - Flow on {u}->{v}: {flow} (capacity: {capacity}) -> {status}")

        print("\n\nResulting path flows for each commodity:")
        for comm, path_flows in solution["path_flows"].items():
            print(f"  - Commodity {comm}:")
            for path, flow in path_flows.items():
                if float(flow) > 1e-6:
                    print(f"    - path {path}: flow = {flow}")

        print("\n\nEquilibrium costs:")
        for route, cost in solution["equilibrium_costs"].items():
            print(f"  - minimized cost for travelers from {route}: {cost}")
    else:
        print("\n\nNo solution found. The safety limits are too strict.")

solve_for_one_network()

while True:
    print("\nAdd a new metro line (or press Enter to exit)")
    src = input("Source station: ").strip()
    if src == "":
        print("Exiting simulation.")
        break
    dst = input("Destination station: ").strip()
    color = input("Line color/name (e.g., blue, yellow): ").strip() or "new"

    # add the new edge both ways
    multigraph_edges.append((src, dst, color))

    print(f"\n Added new metro line: {src} <-> {dst} ({color})")
    print("Recomputing equilibrium with updated network...")
    solve_for_one_network()
