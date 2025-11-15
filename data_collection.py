import json
from formal import *
import itertools

REVENUE_VARIATION_WITH_K_PATH =  "./revenue_variation_with_k.json"
REVENUE_VARIATION_WITH_ROUTE_ADDITION_PATH = "./revenue_variation_with_adding_routes.json"

def init_json(filepath,commodities, metro_lines):
    data = {
        "commodities": commodities,
        "edges": metro_lines,
        "experiments": []
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def append_revenue(k, revenue):
    with open(REVENUE_VARIATION_WITH_K_PATH) as f:
        data = json.load(f)

    data["experiments"].append({
        "k": k,
        "revenue": revenue
    })

    with open(REVENUE_VARIATION_WITH_K_PATH, "w") as f:
        json.dump(data, f, indent=2)

def compute_revenue(solution):
    total_revenue = 0.0
    if solution is None:
        print("no solution")
    else:
        prices = {}

        for (u, v), p in solution["prices"].items():
            prices[(u, v)] = float(p.replace("?", ""))

        for (u, v), flow in solution["edge_flows"].items():
            f = float(flow.replace("?", ""))
            p = prices.get((u, v), 0.0)

            total_revenue += f * p

        print("Total System Revenue:", round(total_revenue,2))
    return round(total_revenue,2)

def add_revenue(k,solution):
    total_revenue = compute_revenue(solution)
    append_revenue(k,total_revenue)

def append_revenue_route_json(order_id,k, src, dst, revenue):

    with open(REVENUE_VARIATION_WITH_ROUTE_ADDITION_PATH, "r") as f:
        data = json.load(f)
    
    if k==1:
        exp = {
            "order_id": order_id,  
            "steps": [],
            "final_revenue": None
        }
        data["experiments"].append(exp)
    exp = None
    for e in data["experiments"]:
        if e["order_id"] == order_id:
            exp = e
            break
    exp["steps"].append({
        "step": k,
        "added_route": { "src": src, "dst": dst },
        "revenue": revenue
    })
    exp["final_revenue"] = revenue

    with open(REVENUE_VARIATION_WITH_ROUTE_ADDITION_PATH, "w") as f:
        json.dump(data, f, indent=4)

def add_revenue_route(order_id,k,src,dst,solution):
    total_revenue = compute_revenue(solution)
    append_revenue_route_json(order_id,k, src, dst, total_revenue)
   
def compute_average_utility(commodities, solution):
    eq_costs = solution.get("equilibrium_costs", {})
    total_utility = 0.0
    total_demand = 0.0

    for s, d, w in commodities:
        route = f"{s}->{d}"
        cost = eq_costs.get(route)

        if cost is None:
            continue

        cost = float(cost.replace("?", ""))
        util = -cost

        total_utility += util * w
        total_demand += w

    avg_utility = total_utility / total_demand if total_demand > 0 else None
    print("Average utility:", avg_utility)


def revenue_variation_with_k():
    with open('network.json', 'r') as f:
        data = json.load(f)
        commodities = data["traffic"]
        metro_lines = data["edges"]
        init_json(REVENUE_VARIATION_WITH_K_PATH,commodities,metro_lines)
    read_from_file()
    for k in [0.1, 0.5, 1, 1.5, 2, 5, 10]:
        set_k(k)
        solution = solve_for_one_network()
        add_revenue(k,solution)

def revenue_variation_with_adding_routes():
    with open('network.json', 'r') as f:
        data = json.load(f)
        commodities = data["traffic"]
        metro_lines = data["edges"]
        init_json(REVENUE_VARIATION_WITH_ROUTE_ADDITION_PATH,commodities,metro_lines)
    routes = [["Rabinder Sarobar", "Kavi Subhash", "blue"],
              ["Dum Dum", "Sealdah", "blue"],
              ["Majherhat","Rabinder Sarobar", "blue"]]
    for order_id, ordering in enumerate(itertools.permutations(routes), start=1):
        read_from_file()
        for i in range(3):
            metro_lines.append([ordering[i][0], ordering[i][1], ordering[i][2]])
            set_multigraph_edges(metro_lines)
            set_k(0.5)
            solution = solve_for_one_network()
            add_revenue_route(order_id,i+1,ordering[i][0],ordering[i][1],solution)

if __name__=="__main__":
    revenue_variation_with_k()
    revenue_variation_with_adding_routes()