import csv
import math

# functions for extracting classical training data from supplied csvs
# used for approximating a function as a quantum circuit to obtain an initial target unitary

# tolerance for comparisons
tol = 1e-9

def data_obj(idx, p1, p2, p3, p4, freq, z, b):
    return {
        "idx": idx,
        "p1": float(p1),
        "p2": float(p2),
        "p3": float(p3),
        "p4": float(p4),
        "freq": float(freq),
        "z": float(z),
        "b": float(b),
    }

def get_data(fname):
    first = True
    with open(fname, newline='') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if first: # skip first row containing headers
                first = False
                continue
            yield data_obj(*row)

def tol_eq(a, b):
    return math.isclose(a, b, rel_tol=tol)

def get_data_for_params(fname, fixed_params):
    # fixed params = dict: name->value
    for row in get_data(fname):
        for name, value in fixed_params.items():
            if not tol_eq(row[name], value):
                break
        else: # for else runs else block only if loop finished normally
            yield row

def extract(rows, key):
    for r in rows:
        yield r[key]