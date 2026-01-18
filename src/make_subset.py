import os
import random
import shutil
import logging
from collections import defaultdict
from datetime import datetime
import csv

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/subset_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CONFIG
IMG_DIR = "E:/Downloads/celebA/img_align_celeba"
IDENTITY_FILE = "resources/annotations/identity_CelebA.txt"
OUT_DIR = "resources/celeba_subset"
K_IDS = 2000
M_PER_ID = 30
SEED = 42
TRAIN_PRCNT = 0.7
VAL_PRCNT = 0.15

USE_SYMLINK = False  # True = rápido e não duplica espaço | False = copia


def safe_link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        logger.debug(f"Arquivo já existe, pulando: {dst}")
        return

    try:
        if USE_SYMLINK:
            os.symlink(os.path.abspath(src), dst)
            logger.debug(f"Symlink criado: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            logger.debug(f"Arquivo copiado: {src} -> {dst}")
    except Exception as e:
        logger.error(f"Erro ao processar arquivo {src} -> {dst}: {e}")
        raise


def main():
    start_time = datetime.now()
    logger.info(f"Iniciando criação do subset em {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Criar diretório de resultados se não existir
    os.makedirs("results", exist_ok=True)

    logger.info("Configuração:")
    logger.info(f"  IMG_DIR: {IMG_DIR}")
    logger.info(f"  IDENTITY_FILE: {IDENTITY_FILE}")
    logger.info(f"  OUT_DIR: {OUT_DIR}")
    logger.info(f"  K_IDS: {K_IDS}")
    logger.info(f"  M_PER_ID: {M_PER_ID}")
    logger.info(f"  SEED: {SEED}")
    logger.info(f"  USE_SYMLINK: {USE_SYMLINK}")

    random.seed(SEED)

    # Verificar se arquivos/diretórios existem
    logger.info("Verificando arquivos de entrada...")
    if not os.path.exists(IDENTITY_FILE):
        logger.error(f"Arquivo de identidades não encontrado: {IDENTITY_FILE}")
        raise FileNotFoundError(f"Arquivo não encontrado: {IDENTITY_FILE}")

    if not os.path.exists(IMG_DIR):
        logger.error(f"Diretório de imagens não encontrado: {IMG_DIR}")
        raise FileNotFoundError(f"Diretório não encontrado: {IMG_DIR}")

    logger.info(f"Arquivo de identidades encontrado: {IDENTITY_FILE}")
    logger.info(f"Diretório de imagens encontrado: {IMG_DIR}")

    # id -> [img1, img2, ...]
    id_to_imgs = defaultdict(list)

    logger.info("Lendo arquivo de identidades...")
    total_lines = 0
    valid_lines = 0

    with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                fn, id_str = line.split()
                id_to_imgs[int(id_str)].append(fn)
                valid_lines += 1

                if line_num % 10000 == 0:
                    logger.info(f"  Processadas {line_num} linhas...")

            except ValueError as e:
                logger.warning(f"Linha inválida {line_num}: {line} - Erro: {e}")
                continue

    logger.info(f"Arquivo processado: {total_lines} linhas totais, {valid_lines} linhas válidas")
    logger.info(f"Total de identidades encontradas: {len(id_to_imgs)}")

    # Estatísticas das identidades
    img_counts = [len(imgs) for imgs in id_to_imgs.values()]
    logger.info(f"Estatísticas de imagens por identidade:")
    logger.info(f"  Mínimo: {min(img_counts)} imagens")
    logger.info(f"  Máximo: {max(img_counts)} imagens")
    logger.info(f"  Média: {sum(img_counts) / len(img_counts):.1f} imagens")

    # filtra ids com pelo menos M fotos
    logger.info(f"Filtrando identidades com pelo menos {M_PER_ID} fotos...")
    candidates = [i for i, imgs in id_to_imgs.items() if len(imgs) >= M_PER_ID]
    logger.info(f"Identidades candidatas (>= {M_PER_ID} fotos): {len(candidates)}")

    if len(candidates) < K_IDS:
        error_msg = f"Poucos ids com >= {M_PER_ID} fotos. Achou só {len(candidates)}, precisa de {K_IDS}."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Selecionando {K_IDS} identidades aleatoriamente...")
    chosen_ids = random.sample(candidates, K_IDS)
    logger.info(f"Identidades selecionadas: {sorted(chosen_ids)[:10]}{'...' if len(chosen_ids) > 10 else ''}")

    # limpa saída (opcional)
    logger.info(f"Criando diretório de saída: {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Iniciando cópia das imagens...")
    total = 0
    processed_ids = 0

    split_csv_path = os.path.join(OUT_DIR, "split.csv")
    csv_file = open(split_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["filepath", "person_id", "split"])

    for person_id in chosen_ids:
        processed_ids += 1
        logger.info(f"Processando identidade {person_id} ({processed_ids}/{len(chosen_ids)})")

        imgs = id_to_imgs[person_id]
        logger.info(f"  Imagens disponíveis para ID {person_id}: {len(imgs)}")

        chosen_imgs = random.sample(imgs, M_PER_ID)
        # random.shuffle(chosen_imgs)
        logger.info(f"  Imagens selecionadas: {len(chosen_imgs)}")

        n_train = int(round(M_PER_ID * TRAIN_PRCNT))
        n_val = max(1, int(round(M_PER_ID * VAL_PRCNT)))

        # garante que soma não estoura
        if n_train + n_val > M_PER_ID - 1:
            n_val = M_PER_ID - n_train - 1
        n_test = M_PER_ID - n_train - n_val

        train_imgs = chosen_imgs[:n_train]
        val_imgs = chosen_imgs[n_train:n_train + n_val]
        test_imgs = chosen_imgs[n_train + n_val:]

        splits = [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]
        logger.info(f"  Divisão das imagens: Train = {TRAIN_PRCNT*100}% ({n_train}), Val = {VAL_PRCNT*100}% ({n_val}), Test = {100-((TRAIN_PRCNT + VAL_PRCNT)*100)}% ({n_test})")

        person_total = 0
        for split_name, split_imgs in splits:
            person_dir = os.path.join(OUT_DIR, split_name, f"person_{person_id}")
            os.makedirs(person_dir, exist_ok=True)
            logger.info(f"  Diretório criado: {person_dir}")

            for fn in split_imgs:
                src = os.path.join(IMG_DIR, fn)
                dst = os.path.join(person_dir, fn)

                if not os.path.exists(src):
                    logger.warning(f"  Arquivo fonte não encontrado: {src}")
                    continue

                safe_link(src, dst)
                total += 1
                person_total += 1

                # caminho relativo fica mais portátil
                relpath = os.path.relpath(dst, start=OUT_DIR)
                csv_writer.writerow([relpath.replace("\\", "/"), person_id, split_name])

        logger.info(f"  Identidade {person_id} processada: {person_total} imagens copiadas")

    csv_file.close()
    logger.info(f"Split CSV salvo em: {split_csv_path}")

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 50)
    logger.info("RESUMO DA CRIAÇÃO DO SUBSET:")
    logger.info(f"  Pasta de saída: {OUT_DIR}")
    logger.info(f"  Identidades processadas: {len(chosen_ids)}")
    logger.info(f"  Imagens totais copiadas: {total}")
    logger.info(f"  Imagens esperadas: {K_IDS * M_PER_ID}")
    logger.info(f"  Método usado: {'Symlink' if USE_SYMLINK else 'Cópia'}")
    logger.info(f"  Tempo total: {duration.total_seconds():.2f}s")
    logger.info("=" * 50)

    print("Subset criado!")
    print("Pasta:", OUT_DIR)
    print("IDs:", len(chosen_ids))
    print("Imagens totais:", total)
    print("Symlink:", USE_SYMLINK)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Erro durante a execução: {e}")
        raise