import subprocess
import time

sample_list = [
    "Bi2Sr2CaCu2O8",  # Bismuth strontium calcium copper oxide (High-temperature superconductor)
    "La2CuO4",        # Lanthanum copper oxide (Parent compound for cuprate superconductors)
    "YBa2Cu3O7",      # Yttrium barium copper oxide (High-temperature superconductor, YBCO)
    "NaAlSi3O8",      # Albite (A feldspar mineral commonly found in igneous rocks)
    "KAlSi3O8",       # Microcline (A potassium-rich feldspar mineral)
    "CaTiO3",         # Calcium titanate (Perovskite structure, used in capacitors and other electronics)
    "BaTiO3",         # Barium titanate (Ferroelectric material used in ceramics and capacitors)
    "Sr2RuO4",        # Strontium ruthenate (Layered perovskite, studied for unconventional superconductivity)
    "FeAs",           # Iron arsenide (Part of iron-based superconductors)
    "Ca5(PO4)3(OH)",  # Hydroxyapatite (Main mineral component of bone and teeth)
]

iteration_times = []

model_path = "/home/hice1/kthakrar3/DiffCSP_kt/pretrained_checkpoints"
save_path = "/home/hice1/kthakrar3/DiffCSP_kt/sample_output"
num_evals = 100

for formula in sample_list:
    start_time = time.time() 
    command = [
        "python", "scripts/sample.py",
        "--model_path", model_path,
        "--save_path", save_path,
        "--formula", formula,
        "--num_evals", str(num_evals)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()  

    iteration_times.append(f"Sample {formula}: {end_time - start_time:.2f} seconds")

# print("Times for each iteration:", iteration_times)
