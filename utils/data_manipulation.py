# Cell 3: Caricamento metadata, aggiunta 'method' e filtraggio secondo specifica tutor
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
from tqdm import tqdm
from pathlib import Path

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# import kagglehub

# Provo a scaricare: se esiste in cache non riscarica e restituisce subito
# path = kagglehub.dataset_download("awsaf49/artifact-dataset")
# print("Path dataset:", path)

DATASET_PATH = '/home/alessandro/.cache/kagglehub/datasets/awsaf49/artifact-dataset/versions/1'



def load_metadata(root=DATASET_PATH):
    csvs = glob.glob(os.path.join(root, '**', 'metadata.csv'), recursive=True)
    dfs = []
    for f in csvs:
        df = pd.read_csv(f)
        df['metadata_dir'] = os.path.dirname(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def calculate_image_variance(image_path, method='numpy'):
    """
    Calcola la varianza di un'immagine.
    
    Args:
        image_path: Percorso dell'immagine
        method: 'numpy' (più veloce) o 'opencv'
    
    Returns:
        float: Varianza dell'immagine, 0.0 se errore
    """
    try:
        if method == 'opencv':
            # Metodo OpenCV (più veloce per immagini grandi)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            return np.var(img.astype(np.float32))
        else:
            # Metodo PIL + numpy (più compatibile)
            with Image.open(image_path) as img:
                # Converti in grayscale se necessario
                if img.mode != 'L':
                    img = img.convert('L')
                img_array = np.array(img, dtype=np.float32)
                return np.var(img_array)
    except Exception as e:
        print(f"Errore nel calcolare varianza per {image_path}: {e}")
        return 0.0

def filter_zero_variance_images(df, variance_threshold=1e-6, batch_size=1000):
    """
    Filtra le immagini con varianza zero o molto bassa.
    
    Args:
        df: DataFrame con colonna 'image_path_full'
        variance_threshold: Soglia minima di varianza
        batch_size: Numero di immagini da processare per batch
    
    Returns:
        tuple: (DataFrame filtrato, lista indici rimossi, statistiche)
    """
    print(f"Inizio filtro varianza zero su {len(df)} immagini...")
    print(f"Soglia varianza: {variance_threshold}")
    
    variances = []
    valid_indices = []
    invalid_paths = []
    
    # Processa in batch per mostrare progresso
    for i in tqdm(range(0, len(df), batch_size), desc="Calcolando varianze"):
        batch_end = min(i + batch_size, len(df))
        batch_df = df.iloc[i:batch_end]
        
        for idx, row in batch_df.iterrows():
            variance = calculate_image_variance(row['image_path_full'])
            variances.append(variance)
            
            if variance > variance_threshold:
                valid_indices.append(idx)
            else:
                invalid_paths.append(row['image_path_full'])
    
    # Crea DataFrame filtrato
    df_filtered = df.loc[valid_indices].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Statistiche
    stats = {
        'total_images': len(df),
        'valid_images': len(df_filtered),
        'removed_images': len(df) - len(df_filtered),
        'removal_percentage': ((len(df) - len(df_filtered)) / len(df)) * 100,
        'variance_stats': {
            'min': np.min(variances),
            'max': np.max(variances),
            'mean': np.mean(variances),
            'std': np.std(variances),
            'median': np.median(variances)
        }
    }
    
    return df_filtered, invalid_paths, stats

def print_variance_stats(stats, method_name):
    """Stampa statistiche del filtro varianza."""
    print(f"\n--- Statistiche Filtro Varianza: {method_name} ---")
    print(f"Immagini totali:     {stats['total_images']:>6}")
    print(f"Immagini valide:     {stats['valid_images']:>6}")
    print(f"Immagini rimosse:    {stats['removed_images']:>6}")
    print(f"Percentuale rimossa: {stats['removal_percentage']:>6.2f}%")
    print(f"Varianza min:        {stats['variance_stats']['min']:>10.2e}")
    print(f"Varianza max:        {stats['variance_stats']['max']:>10.2e}")
    print(f"Varianza media:      {stats['variance_stats']['mean']:>10.2e}")
    print(f"Varianza mediana:    {stats['variance_stats']['median']:>10.2e}")


def filter_data(*methods):
    
    df_ld, df_sd, df_coco_sample = methods

    # 4. NUOVO: Filtra immagini a varianza zero per ogni gruppo
    print("\n" + "="*60)
    print("FILTRO VARIANZA ZERO")
    print("="*60)

    # Filtra latent_diffusion
    df_ld_filtered, invalid_ld, stats_ld = filter_zero_variance_images(df_ld)
    print_variance_stats(stats_ld, "Latent Diffusion")

    # Filtra stable_diffusion
    df_sd_filtered, invalid_sd, stats_sd = filter_zero_variance_images(df_sd)
    print_variance_stats(stats_sd, "Stable Diffusion")

    # Filtra COCO sample
    df_coco_filtered, invalid_coco, stats_coco = filter_zero_variance_images(df_coco_sample)
    print_variance_stats(stats_coco, "COCO Sample")

    return df_ld_filtered, df_sd_filtered, df_coco_filtered

# 1. Carica e prepara il DataFrame
def load_dataframe():
    df = load_metadata()
    df['label'] = (df['target'] > 0).astype(int)
    df['image_path_full'] = df.apply(
        lambda r: os.path.join(r['metadata_dir'], r['image_path']),
        axis=1
    )

    # 2. Aggiungi colonna 'method' estraendo il nome della sottocartella
    df['method'] = df['metadata_dir'].apply(lambda p: Path(p).name)
    print("Metodi disponibili:")
    print(df['method'].value_counts())

    # 3. Filtraggio immagini secondo specifica tutor
    # – tutte quelle in latent_diffusion
    # – tutte quelle in stable_diffusion
    # – 10.000 a caso di quelle in coco/train2017
    df_ld = df[df['method'] == 'latent_diffusion']
    df_sd = df[df['method'] == 'stable_diffusion']

    # seleziona solo COCO in train2017
    df_coco_full = df[
            (df['method'] == 'coco') &
            (df['image_path_full'].str.contains('train2017', regex=False))
    ]

    if len(df_coco_full) < 10000:
        raise ValueError(f"Trovate solo {len(df_coco_full)} immagini COCO train2017 (<10000)")

    # sotto-campiona 10k COCO train2017 in modo riproducibile
    df_coco_sample = df_coco_full.sample(n=10000, random_state=seed)

    df_ld_filtered, df_sd_filtered, df_coco_filtered = filter_data(df_ld, df_sd, df_coco_sample)    


    # 5. Ricompone il DataFrame finale con immagini filtrate
    df = pd.concat([df_ld_filtered, df_sd_filtered, df_coco_filtered], ignore_index=True)

    # Report finale
    print(f"\n" + "="*60)
    print("REPORT FINALE")
    print("="*60)
    print(f"Selezionate dopo filtro varianza:")
    print(f"  - Latent Diffusion:  {len(df_ld_filtered):>6} / {len(df_ld):>6} originali")
    print(f"  - Stable Diffusion:  {len(df_sd_filtered):>6} / {len(df_sd):>6} originali")
    print(f"  - COCO (train2017):  {len(df_coco_filtered):>6} / {len(df_coco_sample):>6} originali")
    print(f"  - TOTALE:            {len(df):>6}")

    # Statistiche complessive rimozioni
    total_removed = (len(df_ld) - len(df_ld_filtered) + 
                    len(df_sd) - len(df_sd_filtered) + 
                    len(df_coco_sample) - len(df_coco_filtered))
    original_total = len(df_ld) + len(df_sd) + len(df_coco_sample)
    removal_percent = (total_removed / original_total) * 100

    print(f"\nImmagini rimosse per varianza zero: {total_removed} ({removal_percent:.2f}%)")

    # Dimensioni del DataFrame
    print(f"\nForma del DataFrame finale: {df.shape}")

    # Informazioni sulle colonne e tipi di dati
    print(df.info())

    # Statistiche descrittive
    print(df.describe())
    print(df.head())
    print(df.tail())

    # Reset indice
    df = df.reset_index(drop=True)

    return df

def organize_images_with_symlinks(df, output_base_dir="cache/data_subsets"):
    """
    Organizza le immagini del dataframe in cartelle separate per metodo usando link simbolici.
    
    Args:
        df: DataFrame con colonne 'method', 'image_path_full'
        output_base_dir: Cartella base dove creare le sottocartelle
    """
    
    # Crea la cartella base se non esiste
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Ottieni i metodi unici
    methods = df['method'].unique()
    print(f"Metodi trovati: {methods}")
    
    # Statistiche per verificare
    print(f"\nDistribuzione immagini per metodo:")
    print(df['method'].value_counts())
    
    # Contatori per il tracking
    linked_files = {method: 0 for method in methods}
    errors = []
    
    # Processa per ogni metodo
    for method in methods:
        # Crea cartella di destinazione
        dest_dir = os.path.join(output_base_dir, method)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Filtra immagini per questo metodo
        method_df = df[df['method'] == method]
        
        print(f"\n--- Processando {method} ({len(method_df)} immagini) ---")
        
        # Crea link simbolici con progress bar
        for idx, row in tqdm(method_df.iterrows(), 
                           total=len(method_df), 
                           desc=f"Linkando {method}"):
            
            source_path = row['image_path_full']
            
            # Verifica che il file sorgente esista
            if not os.path.exists(source_path):
                errors.append(f"File non trovato: {source_path}")
                continue
            
            # Nome del file di destinazione
            filename = os.path.basename(source_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # Gestisce duplicati aggiungendo un suffisso
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path) or os.path.islink(dest_path):
                name, ext = os.path.splitext(original_dest_path)
                dest_path = f"{name}_{counter}{ext}"
                counter += 1
            
            try:
                # Crea link simbolico assoluto
                source_abs = os.path.abspath(source_path)
                os.symlink(source_abs, dest_path)
                linked_files[method] += 1
                
            except Exception as e:
                errors.append(f"Errore linkando {source_path}: {str(e)}")
    
    # Report finale
    print(f"\n{'='*50}")
    print("REPORT FINALE:")
    print(f"{'='*50}")
    
    total_linked = sum(linked_files.values())
    
    for method, count in linked_files.items():
        expected = len(df[df['method'] == method])
        print(f"{method:20}: {count:>6} / {expected:>6} linkate")
    
    print(f"{'='*50}")
    print(f"TOTALE LINKATE:      {total_linked:>6} / {len(df):>6}")
    
    if errors:
        print(f"\nERRORI ({len(errors)}):")
        for error in errors[:10]:  # Mostra solo i primi 10 errori
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... e altri {len(errors) - 10} errori")
    
    folders = []
    cwd = os.path.abspath(output_base_dir)
    print(f"\nCartelle create in: {cwd}")
    for method in methods:
        folders.append(cwd + f"/{method}")
        method_dir = os.path.join(output_base_dir, method)
        file_count = len([f for f in os.listdir(method_dir) 
                         if os.path.isfile(os.path.join(method_dir, f)) or 
                            os.path.islink(os.path.join(method_dir, f))])
        print(f"  - {method}/: {file_count} link simbolici")

    return folders
