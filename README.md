# Tradutor Local Windows — pipeline concorrente, frases coerentes e saída em streaming

Tradutor local de áudio do navegador para voz pt-BR: o Whisper reconhece a fala em inglês, o Ollama/TranslateGemma traduz para português brasileiro e o Kokoro sintetiza a voz, tudo localmente.

## Como o áudio flui

```unknown
Navegador (vídeo)  →  CABLE Input  →  tradutor captura CABLE Output  →  tradução  →  FxSound  →  headset
```

1. **Chrome/Edge**: no Mixer de volume do Windows (Configurações → Sistema → Som → Mixer de volume), defina a saída do navegador como `CABLE Input (VB-Audio Virtual Cable)`. Os demais aplicativos continuam na saída normal.
2. **VB-CABLE**: une `CABLE Input` e `CABLE Output`.
3. **Tradutor**: captura `CABLE Output` automaticamente, traduz e reproduz na saída escolhida com `--output`.
4. **FxSound**: recebe o áudio traduzido e envia ao headset. Não altere o FxSound para apontar para o CABLE Input.

## Recursos usados (configuração atual)

| Componente | Onde roda | Recurso |
|---|---|---|
| Whisper (ASR) `large-v3` | **GPU** NVIDIA (CUDA, FP16) | ~3,9 GB de VRAM |
| Ollama / TranslateGemma `4b` | **GPU** NVIDIA | ~2,9 GB de VRAM |
| Kokoro (TTS) | **CPU** | até 8 threads intra-op por worker |
| RAM | — | ~32 GB com folga |

Na RTX 5060 Ti (16 GB de VRAM), o conjunto completo usa ~10 GB, sobrando ~6 GB. O Kokoro roda **propositalmente na CPU**: sintetiza ~2,7x mais rápido que o tempo real com 8 threads, e isso evita disputar a GPU com o Whisper e o Ollama, que são os verdadeiros gargalos.

O Whisper é opcional entre `base`, `small`, `medium` e `large-v3` (`--whisper-model`). Ordem de tamanho/precisão: `tiny < base < small < medium < large-v3`. O `large-v3` transcreve 4 s de áudio em ~0,4 s na GPU.

## Instalação do zero

Pré-requisitos de sistema (instalados uma vez):

- Python 3.12 e o ambiente virtual `.venv` (crie com `python -m venv .venv`).
- VB-CABLE instalado (driver do cabo virtual).
- Ollama instalado e rodando, com o modelo de tradução baixado:
  ```powershell
  ollama pull translategemma:4b
  ```
- eSpeak NG instalado (usado pela biblioteca misaki para o G2P pt-BR).
- `kokoro-v1.0.onnx` e `voices-v1.0.bin` ao lado de `tradutor_local.py`.

Dependências Python (dentro do `.venv`):

```powershell
cd D:\Projetos\tradutor-local-windows-repaired
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O `requirements.txt` inclui `faster-whisper`, `kokoro-onnx`, `misaki`, `soundcard`, `sounddevice`, `scipy`, `requests`, `onnxruntime` e as DLLs de GPU `nvidia-cublas-cu12` + `nvidia-cudnn-cu12`.

### GPU: o segredo das DLLs

O `faster-whisper` só usa a GPU se as DLLs de **cuBLAS (CUDA 12)** e **cuDNN 9** estiverem acessíveis no `PATH`. Nesta instalação elas vêm via pip (sem instalar o CUDA Toolkit inteiro):

```
.venv\Lib\site-packages\nvidia\cublas\bin   →  cublas64_12.dll
.venv\Lib\site-packages\nvidia\cudnn\bin    →  cudnn64_9.dll
.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin
```

O `iniciar.bat` já adiciona esses três diretórios ao `PATH` antes de executar. O modo `--whisper-device auto` testa uma inferência real de CUDA na inicialização; se as DLLs não estiverem acessíveis, ele recua para CPU INT8 antes de iniciar a captura (com aviso explícito).

## Iniciar a tradução

Dê dois cliques em **`iniciar.bat`** (ou digite `iniciar` no terminal dentro da pasta). Ele configura o PATH das DLLs e roda:

```powershell
.\.venv\Scripts\python.exe -u .\tradutor_local.py --language en --whisper-model large-v3 --output "FxSound Audio Enhancer"
```

Para executar manualmente com outras opções:

```powershell
cd D:\Projetos\tradutor-local-windows-repaired
.\.venv\Scripts\python.exe -u .\tradutor_local.py --language en --whisper-model large-v3 --output "FxSound Audio Enhancer"
```

Se não ouvir no FxSound, use o headset diretamente:

```powershell
.\.venv\Scripts\python.exe -u .\tradutor_local.py --language en --whisper-model large-v3 --output "H510-PRO Wireless headset"
```

## Testar a saída contínua

```powershell
.\.venv\Scripts\python.exe .\tradutor_local.py --test-stream-output --output "FxSound Audio Enhancer"
```

Se o teste não for audível no FxSound, tente diretamente no headset:

```powershell
.\.venv\Scripts\python.exe .\tradutor_local.py --test-stream-output --output "H510-PRO Wireless headset"
```

## Ajustes e diagnósticos

```powershell
.\.venv\Scripts\python.exe .\tradutor_local.py --list-devices
.\.venv\Scripts\python.exe .\tradutor_local.py --test-translation
.\.venv\Scripts\python.exe .\tradutor_local.py --monitor-input
```

Observe no log:

- `[MODELO] Whisper pronto: dispositivo=cuda, precisão=float16...`: o ASR está na GPU. A linha `Kokoro: CPUExecutionProvider` é intencional (TTS na CPU).
- `[AVISO] GPU para Whisper não passou no teste; fallback para CPU INT8...`: as DLLs de cuBLAS/cuDNN não estão acessíveis no PATH; rode pelo `iniciar.bat` ou ajuste o PATH.
- `[ORIGINAL/en]`: texto reconhecido e montado em enunciados.
- `[PORTUGUÊS #id.parte]`: unidade de tradução entregue ao sintetizador.
- `[TEMPO] ASR/segmentação`: quanto áudio de origem foi agregado antes de seguir à tradução.
- `[LATÊNCIA]`: tempo desde o começo da fala de origem até o primeiro bloco de voz audível.
- `[TEMPO REAL] ... descartado`: o programa descartou trabalho velho para evitar acumular atraso indefinidamente.

### Como o pipeline funciona

- **Teste real de CUDA no início:** em `--whisper-device auto`, o programa executa uma inferência curta antes de começar a captura. Se CUDA/cuBLAS/cuDNN falhar, recua para CPU INT8 imediatamente.
- **Janelas ASR de até 2,0 s** com silêncio de 0,40 s marcando uma pausa provável de frase.
- **Montagem de enunciados sem cortar a frase no meio:** janelas que parecem continuações são acumuladas; ao liberar um enunciado (pausa, limite de contexto ou pontuação), ele é cortado apenas na última fronteira de frase concluída, e a cauda que ainda está falando é carregada para o enunciado seguinte. Fragmentos de uma ou duas palavras ficam retidos em vez de serem traduzidos sozinhos.
- **Pausa no meio de frase respeitada:** se uma janela termina numa pausa curta mas o texto ainda não terminou a frase, o programa aguarda até ~2,2 s por uma continuação antes de se comprometer; se o falante retomar, a frase segue inteira. Isso evita a sensação de frase cortada ao meio.
- **Tradução em streaming coerente:** o texto traduzido é liberado em limites de frase; para frases muito longas sem pontuação, a divisão só acontece numa fronteira forte de cláusula (`;`, `:`, `—`, `,`) — nunca no meio da frase. A saída preserva pontuação de frase/cláusula para o TTS manter o mesmo ritmo.
- **Pausas naturais na voz:** o Kokoro insere pausa adequada após ponto (0,32 s) e após vírgula/ponto-e-vírgula (0,15 s), e há uma pausa controlada entre partes faladas, então a tradução não sai "cortada" nem com emendas secas.
- **Compromisso antecipado em janela cheia:** quando uma janela cheia termina com uma frase já completa, essa frase é enviada à tradução imediatamente, em vez de esperar toda a frase longa terminar — reduzindo a defasagem entre o vídeo e o áudio.
- **Pausa sem cortar a frase em andamento:** ao detectar silêncio prolongado, a frase que já começou a tocar termina e novas falas antigas são bloqueadas; na retomada, trechos capturados antes são descartados.
- **Buffers limitados e ordem mantida:** ASR, tradução e TTS são etapas concorrentes com filas curtas e voz reproduzida em ordem.

## Limitação real de latência

O pipeline reduz trechos fragmentados e evita cortar a frase em reprodução, mas ainda não é um ASR token-a-token. O Whisper trabalha sobre janelas curtas e o programa monta enunciados estáveis entre elas, comprometendo cada frase concluída assim que ela aparece — em vez de esperar toda a fala contínua terminar. A baixa latência final depende de o reconhecimento, o TranslateGemma e o Kokoro acompanharem a velocidade da fala. Com o ASR `large-v3` na GPU, a transcrição é da ordem de 0,3–0,4 s por janela de 2,0 s; a latência final varia com o modelo e a carga, sem garantia de sub-segundo.

Dicas para reduzir a defasagem:
- Use o modelo ASR menor (`--whisper-model small` ou `medium`) se aceitar um pouco menos de precisão — a transcrição fica bem mais rápida.
- Se o Ollama suportar, defina `OLLAMA_NUM_PARALLEL=2` (ou 3) e reinicie o servidor: várias frases passam a ser traduzidas em paralelo, evitando "buracos" entre frases.
- `--tts-speed` acima de 1.2 deixa a fala um pouco mais rápida (padrão já é 1.2).
- Ajuste `--asr-workers`/`--tts-workers` para o seu hardware; em CPU fraco, workers extras disputam recursos e podem aumentar a latência.

Para uma próxima etapa realmente simultânea, vale substituir o segmentador caseiro por um ASR com política de compromisso de texto estável, como Whisper-Streaming/SimulStreaming, e medir separadamente os tempos de ASR, tradução e TTS.

## Custo e privacidade

Depois dos downloads iniciais, o reconhecimento, a tradução e a voz são executados localmente. O único endpoint do tradutor é o Ollama local em `127.0.0.1:11434`; não são usadas APIs pagas.#   t r a d u t o r - l o c a l  
 #   t r a d u t o r - l o c a l  
 