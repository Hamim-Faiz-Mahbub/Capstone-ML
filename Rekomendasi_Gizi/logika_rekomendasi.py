import pandas as pd
import random
import numpy as np

# <-- PERUBAHAN: Menambahkan 'fiber_g' ke dalam data AKG
DATA_AKG_PMK_2019 = [
    (10, 12, 'semua', {'energy_kal': 2000, 'protein_g': 50, 'fat_g': 65, 'carbohydrate_g': 300, 'fiber_g': 28}),
    (13, 15, 'laki-laki', {'energy_kal': 2400, 'protein_g': 65, 'fat_g': 80, 'carbohydrate_g': 350, 'fiber_g': 34}),
    (13, 15, 'perempuan', {'energy_kal': 2050, 'protein_g': 65, 'fat_g': 70, 'carbohydrate_g': 300, 'fiber_g': 29}),
    (16, 18, 'laki-laki', {'energy_kal': 2650, 'protein_g': 75, 'fat_g': 85, 'carbohydrate_g': 400, 'fiber_g': 37}),
    (16, 18, 'perempuan', {'energy_kal': 2100, 'protein_g': 65, 'fat_g': 70, 'carbohydrate_g': 300, 'fiber_g': 29}),
    (19, 29, 'laki-laki', {'energy_kal': 2650, 'protein_g': 65, 'fat_g': 75, 'carbohydrate_g': 430, 'fiber_g': 38}),
    (19, 29, 'perempuan', {'energy_kal': 2250, 'protein_g': 60, 'fat_g': 65, 'carbohydrate_g': 360, 'fiber_g': 32}),
    (30, 49, 'laki-laki', {'energy_kal': 2550, 'protein_g': 65, 'fat_g': 70, 'carbohydrate_g': 415, 'fiber_g': 36}),
    (30, 49, 'perempuan', {'energy_kal': 2150, 'protein_g': 60, 'fat_g': 60, 'carbohydrate_g': 340, 'fiber_g': 30}),
    (50, 64, 'laki-laki', {'energy_kal': 2150, 'protein_g': 65, 'fat_g': 60, 'carbohydrate_g': 340, 'fiber_g': 30}),
    (50, 64, 'perempuan', {'energy_kal': 1800, 'protein_g': 60, 'fat_g': 50, 'carbohydrate_g': 280, 'fiber_g': 25}),
    (65, 80, 'laki-laki', {'energy_kal': 1800, 'protein_g': 64, 'fat_g': 50, 'carbohydrate_g': 275, 'fiber_g': 25}),
    (65, 80, 'perempuan', {'energy_kal': 1550, 'protein_g': 58, 'fat_g': 45, 'carbohydrate_g': 235, 'fiber_g': 22}),
    (80, 999, 'laki-laki', {'energy_kal': 1600, 'protein_g': 64, 'fat_g': 45, 'carbohydrate_g': 235, 'fiber_g': 22}),
    (80, 999, 'perempuan', {'energy_kal': 1400, 'protein_g': 58, 'fat_g': 40, 'carbohydrate_g': 205, 'fiber_g': 20})
]
BATAS_GULA_HARIAN = [ (0, 1, 0), (2, 4, 16), (4, 7, 20), (7, 10, 23), (10, 13, 27), (13, 15, 32), (18, 150, 50) ]

def get_full_day_targets(umur, jenis_kelamin):
    jk = jenis_kelamin.lower()
    for min_age, max_age, gender, targets in DATA_AKG_PMK_2019:
        if gender == 'semua' or jk.startswith(gender[0]):
            if umur >= min_age and umur < max_age + 1:
                final_targets = targets.copy()
                final_targets['sodium_mg_limit'] = 1500.0
                batas_gula = 50.0
                for min_a, max_a, limit in BATAS_GULA_HARIAN:
                    if umur >= min_a and umur <= max_a:
                        batas_gula = float(limit)
                        break
                final_targets['sugar_g_limit'] = batas_gula
                return final_targets
    return None

def calculate_remaining_needs(full_targets, consumed):
    remaining = {}
    for nutrient_key, target_value in full_targets.items():
        consumed_key = nutrient_key.replace('_limit', '')
        consumed_value = consumed.get(consumed_key, 0)
        remaining[nutrient_key] = max(0, target_value - consumed_value)
    return remaining

def generate_recommendations(df, needs, num_prod=3, iters=30000):
    best_combo, best_score = None, float('inf')
    sisa_gula_limit, sisa_sodium_limit = needs.get('sugar_g_limit', 0), needs.get('sodium_mg_limit', 0)
    for _ in range(iters):
        combo = df.sample(n=num_prod) if len(df) >= num_prod else df
        rekom_gula, rekom_sodium = combo['total_sugar_g'].sum(), combo['total_sodium_mg'].sum()
        if rekom_gula > sisa_gula_limit or rekom_sodium > sisa_sodium_limit:
            continue

        nutr = {'energy_kal': combo['total_energy_kal'].sum(), 'protein_g': combo['total_protein_g'].sum(),
                'fat_g': combo['total_fat_g'].sum(), 'carbohydrate_g': combo['total_carbohydrate_g'].sum(),
                'fiber_g': combo['total_fiber_g'].sum()} # <-- PERUBAHAN

        w_kal, w_makro, w_batas = 0.5, 0.4, 0.1
        score_kal = abs(nutr['energy_kal'] - needs['energy_kal']) / needs['energy_kal'] if needs['energy_kal'] > 0 else 0
        
        # <-- PERUBAHAN: Menambahkan serat ke dalam perhitungan skor makro
        score_pro = abs(nutr['protein_g'] - needs['protein_g']) / needs['protein_g'] if needs['protein_g'] > 0 else 0
        score_fat = abs(nutr['fat_g'] - needs['fat_g']) / needs['fat_g'] if needs['fat_g'] > 0 else 0
        score_car = abs(nutr['carbohydrate_g'] - needs['carbohydrate_g']) / needs['carbohydrate_g'] if needs['carbohydrate_g'] > 0 else 0
        score_fib = abs(nutr['fiber_g'] - needs['fiber_g']) / needs['fiber_g'] if needs.get('fiber_g', 0) > 0 else 0
        score_makro_avg = (score_pro + score_fat + score_car + score_fib) / 4

        score_gula = (rekom_gula / sisa_gula_limit) if sisa_gula_limit > 0 else (1 if rekom_gula > 0 else 0)
        score_sod = (rekom_sodium / sisa_sodium_limit) if sisa_sodium_limit > 0 else (1 if rekom_sodium > 0 else 0)
        score_limit = (score_gula + score_sod) / 2
        
        total_score = (score_kal * w_kal) + (score_makro_avg * w_makro) + (score_limit * w_batas)
        
        if total_score < best_score:
            best_score, best_combo = total_score, combo
            
    return best_combo

print("File 'logika_rekomendasi.py' (dengan logika serat) berhasil dibuat.")
