# libkser - Serenity Serialization Format v1

Bibliothèque C cross-platform pour la sérialisation des modèles k-mamba.

## Format .ser v1

```
[16 bytes header] "SERENITY" + η (0xCE 0xB7) + version + reserved
[Config block]    Model architecture (vocab_size, dim, layers, etc.)
[Tensor index]    Table des matières des tensors
[Vocab block]     Vocabulary embarqué (si vocab_size > 0)
[Tensor data]     Poids du modèle (FP32/FP16/BF16/INT8)
[SHA256]          32 bytes checksum
```

## Build

```bash
make                    # lib statique + dynamique
make test              # Tests
make install           # Installation système
```

## API

### Écriture

```c
KSerConfig cfg = {
    .vocab_size = 100277,
    .dim = 3072,
    .n_layers = 24,
    .dtype = KSER_BF16
};

KSerWriter* w = kser_writer_create("model.ser", &cfg);
kser_writer_add_vocab(w, 0, "hello", 5);
kser_writer_add_tensor(w, "embedding", weights, shape, KSER_FP32, KSER_BF16);
kser_writer_finalize(w);
kser_writer_free(w);
```

### Lecture

```c
KSerReader* r = kser_reader_open("model.ser");
const KSerConfig* cfg = kser_reader_config(r);
float* weights = kser_reader_load_tensor(r, "embedding", KSER_FP32);
kser_reader_close(r);
```

### Info rapide

```c
KSerInfo info = kser_file_info("model.ser");
printf("Model: %s, Params: %.1fB\n", info.model_name, info.n_params / 1e9);
```

## Quantization

- **FP32** → FP32: copie directe
- **FP32** → FP16: IEEE 754 half-precision
- **FP32** → BF16: Brain Float 16 (Google TPU)
- **FP32** → INT8: min-max per-tensor quantization

## License

Privé - Usage interne k-mamba uniquement.
