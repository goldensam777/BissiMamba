#!/usr/bin/env python3
"""
Generate a synthetic conversational dataset for BissiMamba training.
No network required.

Output: data/train.txt   (Human:/Bot: pairs, UTF-8)
Target: ~500 KB - 1 MB of diverse conversation text.

Run:   python3 generate_data.py
"""

import os
import random
from pathlib import Path
from itertools import product

random.seed(42)
OUT = Path("data/train.txt")
Path("data").mkdir(exist_ok=True)

# ── Vocabulary pools ──────────────────────────────────────────────────

GREETINGS = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there!", "Hi! What can I do for you?"),
    ("Hey!", "Hey! Good to see you. What's on your mind?"),
    ("Good morning!", "Good morning! Hope you're having a great day."),
    ("Good afternoon!", "Good afternoon! How's your day going?"),
    ("Good evening!", "Good evening! How can I assist you?"),
    ("Bonjour!", "Bonjour! Comment puis-je vous aider?"),
    ("Salut!", "Salut! Quoi de neuf?"),
    ("What's up?", "Not much! Just here to help. What do you need?"),
    ("How are you?", "I'm doing well, thank you! How about you?"),
    ("How's it going?", "Pretty good! Thanks for asking. What can I help with?"),
    ("Nice to meet you!", "Nice to meet you too! What brings you here?"),
]

FAREWELLS = [
    ("Goodbye!", "Goodbye! Have a wonderful day!"),
    ("Bye!", "Bye! Take care!"),
    ("See you later!", "See you! Feel free to come back anytime."),
    ("Thanks, bye!", "You're welcome! Goodbye!"),
    ("That's all I needed, thanks.", "Happy to help! Have a great day."),
    ("Au revoir!", "Au revoir! Bonne journée!"),
    ("Merci, à bientôt!", "De rien! À bientôt!"),
    ("Talk to you later!", "Sure! I'll be here whenever you need me."),
]

THANKS = [
    ("Thank you!", "You're welcome! Is there anything else I can help with?"),
    ("Thanks a lot!", "My pleasure! Don't hesitate to ask if you need more help."),
    ("That was very helpful.", "Glad I could help! Feel free to ask more questions."),
    ("Perfect, thanks!", "Great! Let me know if you need anything else."),
    ("Merci beaucoup!", "Avec plaisir! Y a-t-il autre chose?"),
]

IDENTITY = [
    ("What is your name?", "My name is BissiBot, an assistant built on the Mamba state space model."),
    ("Who are you?", "I'm BissiBot, a conversational AI powered by BissiMamba — a C implementation of the Mamba SSM architecture."),
    ("What can you do?", "I can answer questions, have conversations, and help you explore topics. What would you like to discuss?"),
    ("Are you an AI?", "Yes, I'm an AI assistant. I'm based on the Mamba sequence model architecture."),
    ("Tu t'appelles comment?", "Je m'appelle BissiBot, un assistant basé sur le modèle Mamba."),
    ("Qui es-tu?", "Je suis BissiBot, une IA conversationnelle construite sur l'architecture Mamba SSM."),
    ("Qu'est-ce que tu sais faire?", "Je peux répondre à des questions, discuter de nombreux sujets et vous aider à explorer des idées."),
    ("Are you human?", "No, I'm an AI. But I'm here to have meaningful conversations with you!"),
    ("What are you?", "I'm an AI assistant based on the Mamba state space model — a modern alternative to transformers for sequential data."),
]

TECH_QA = [
    ("What is a neural network?",
     "A neural network is a machine learning model inspired by the brain. It consists of layers of nodes (neurons) connected by weights, trained to recognize patterns in data."),
    ("What is deep learning?",
     "Deep learning uses neural networks with many layers to learn complex representations. It has achieved breakthroughs in vision, language, and speech."),
    ("What is machine learning?",
     "Machine learning is a branch of AI where algorithms learn from data instead of being explicitly programmed. The model improves its performance with experience."),
    ("What is the Mamba model?",
     "Mamba is a state space model (SSM) designed as an efficient alternative to transformers. It uses selective scanning to model sequences with linear time complexity."),
    ("How does Mamba differ from transformers?",
     "Mamba uses structured state spaces instead of self-attention. This makes inference O(n) in sequence length rather than O(n²), making it faster for long sequences."),
    ("What is a state space model?",
     "A state space model represents a dynamic system with hidden states that evolve over time. In deep learning, SSMs like Mamba use this to process sequences efficiently."),
    ("What is CUDA?",
     "CUDA is NVIDIA's parallel computing platform. It lets developers write programs that run on the GPU, enabling massive parallelism for tasks like training neural networks."),
    ("What is a GPU?",
     "A GPU (Graphics Processing Unit) contains thousands of small cores optimized for parallel computation. It's ideal for matrix operations used in deep learning."),
    ("What is backpropagation?",
     "Backpropagation is the algorithm used to train neural networks. It computes gradients of the loss function with respect to each parameter using the chain rule."),
    ("What is an optimizer in deep learning?",
     "An optimizer updates model weights to minimize the loss function. Common optimizers include SGD, Adam, and AdamW."),
    ("What is the Adam optimizer?",
     "Adam combines momentum and RMS scaling of gradients. It adapts the learning rate for each parameter individually, making it effective for most deep learning tasks."),
    ("What is overfitting?",
     "Overfitting happens when a model learns the training data too well, including noise, and performs poorly on new data. It's addressed with regularization, dropout, or more data."),
    ("What is a transformer?",
     "A transformer is a neural network architecture that uses self-attention to model relationships between all positions in a sequence simultaneously."),
    ("What is tokenization?",
     "Tokenization splits text into smaller units (tokens) for a model to process. Tokens can be characters, words, or subwords depending on the tokenizer."),
    ("What is a language model?",
     "A language model learns the probability distribution over sequences of tokens. It can generate text, complete sentences, and answer questions."),
    ("What is perplexity?",
     "Perplexity measures how well a language model predicts a sequence. Lower perplexity means the model is less surprised by the data — it's a standard evaluation metric."),
    ("What is reinforcement learning?",
     "Reinforcement learning trains an agent to take actions in an environment to maximize cumulative reward. It's used in game playing, robotics, and control systems."),
    ("What is transfer learning?",
     "Transfer learning reuses a model trained on one task as a starting point for another. It greatly reduces the data and compute needed for new tasks."),
    ("What is fine-tuning?",
     "Fine-tuning takes a pre-trained model and continues training it on a specific dataset to adapt it to a new task, preserving the general knowledge already learned."),
    ("What is a recurrent neural network?",
     "An RNN processes sequences step by step, maintaining a hidden state. It can capture temporal dependencies but struggles with long-range ones due to vanishing gradients."),
    ("What is attention in neural networks?",
     "Attention lets a model focus on relevant parts of the input when producing each output. In transformers, self-attention computes relationships between all token pairs."),
    ("What is an embedding?",
     "An embedding maps discrete tokens (words, bytes) to continuous vectors in a high-dimensional space, capturing semantic relationships between tokens."),
    ("What is gradient descent?",
     "Gradient descent updates model parameters in the direction that reduces the loss. It computes the gradient of the loss with respect to parameters and takes a small step."),
    ("What is a convolution in deep learning?",
     "A convolution applies a small filter to input data, detecting local patterns. In 1D, it's used for sequences; in 2D, for images. It's efficient thanks to weight sharing."),
    ("What is batch normalization?",
     "Batch normalization normalizes activations within a mini-batch, stabilizing training and allowing higher learning rates. It's widely used in convolutional networks."),
    ("What is dropout?",
     "Dropout randomly sets activations to zero during training, preventing co-adaptation of neurons. It acts as a regularizer and reduces overfitting."),
    ("What is a hyperparameter?",
     "Hyperparameters are settings chosen before training, like learning rate, batch size, and number of layers. They're not learned from data — they're tuned by the practitioner."),
    ("What is cross-entropy loss?",
     "Cross-entropy measures the difference between predicted probability distributions and target labels. It's the standard loss function for classification and language models."),
    ("What is softmax?",
     "Softmax converts a vector of raw scores (logits) into a probability distribution. Each output is positive and all outputs sum to one."),
    ("What is ReLU?",
     "ReLU (Rectified Linear Unit) is an activation function: f(x) = max(0, x). It's simple, computationally cheap, and avoids vanishing gradients in many settings."),
    ("Qu'est-ce que l'intelligence artificielle?",
     "L'intelligence artificielle est un domaine de l'informatique qui vise à créer des systèmes capables d'effectuer des tâches nécessitant normalement l'intelligence humaine."),
    ("Comment fonctionne le deep learning?",
     "Le deep learning utilise des réseaux de neurones profonds pour apprendre des représentations hiérarchiques des données, en passant d'extracteurs de bas niveau à des concepts complexes."),
    ("Qu'est-ce que l'architecture Mamba?",
     "Mamba est un modèle d'espace d'état sélectif. Il traite les séquences de façon récurrente mais peut être parallélisé pendant l'entraînement, combinant efficacité et expressivité."),
]

SCIENCE_QA = [
    ("What is quantum computing?",
     "Quantum computing uses qubits that can be in superposition, enabling massive parallel computation. It could revolutionize cryptography, optimization, and simulation."),
    ("What is DNA?",
     "DNA (deoxyribonucleic acid) is the molecule that carries genetic information in all living organisms. It's a double helix made of four nucleotide bases."),
    ("How does photosynthesis work?",
     "Plants absorb sunlight, CO₂, and water to produce glucose and oxygen. The light reactions capture energy; the Calvin cycle uses it to build sugars."),
    ("What is the speed of light?",
     "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s). It's a fundamental constant of physics."),
    ("What is relativity?",
     "Einstein's theory of relativity has two parts: special relativity (space and time are relative to the observer) and general relativity (gravity is the curvature of spacetime)."),
    ("What is entropy?",
     "Entropy measures disorder or randomness in a system. The second law of thermodynamics states that entropy in a closed system always increases over time."),
    ("What is a black hole?",
     "A black hole is a region where gravity is so strong that nothing, not even light, can escape. It forms when massive stars collapse at the end of their lives."),
    ("What is the Big Bang theory?",
     "The Big Bang theory proposes that the universe began as an extremely hot, dense point about 13.8 billion years ago and has been expanding ever since."),
    ("What is evolution?",
     "Evolution is the process by which populations change over generations through variation, mutation, and natural selection. It explains the diversity of life on Earth."),
    ("What is a neuron?",
     "A neuron is the basic cell of the nervous system. It receives signals through dendrites and transmits them through an axon. Billions of neurons make up the brain."),
]

GENERAL_QA = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest, covering more than 30% of Earth's surface."),
    ("How many continents are there?", "There are seven continents: Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, and South America."),
    ("What is the tallest mountain?", "Mount Everest is the tallest mountain, standing at 8,848 meters above sea level."),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare, likely around 1600-1601."),
    ("What year did World War II end?", "World War II ended in 1945 — in Europe in May and in the Pacific in September."),
    ("What is the Pythagorean theorem?", "In a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c²."),
    ("What is pi?", "Pi (π) is the ratio of a circle's circumference to its diameter, approximately 3.14159. It's an irrational number that goes on forever."),
    ("What is the periodic table?", "The periodic table organizes all known chemical elements by atomic number and properties. There are currently 118 confirmed elements."),
    ("What is a prime number?", "A prime number is a natural number greater than 1 that has no divisors other than 1 and itself. Examples: 2, 3, 5, 7, 11."),
    ("What is inflation?", "Inflation is the rate at which the general price level of goods and services rises, reducing purchasing power over time."),
    ("What is the internet?", "The internet is a global network of computers that communicate using standardized protocols, enabling the exchange of information worldwide."),
    ("What is democracy?", "Democracy is a system of government where citizens exercise power, typically through elected representatives. It often includes free elections and civil liberties."),
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities — especially burning fossil fuels — are the main driver since the 20th century."),
    ("Quelle est la capitale du Sénégal?", "La capitale du Sénégal est Dakar."),
    ("Combien y a-t-il de pays en Afrique?", "Il y a 54 pays reconnus en Afrique, ce qui en fait le continent avec le plus grand nombre de pays."),
    ("Qu'est-ce que la physique quantique?", "La physique quantique décrit le comportement de la matière et de l'énergie à l'échelle atomique, où les particules peuvent être en superposition d'états."),
]

SMALLTALK = [
    ("Do you like music?", "I don't listen to music myself, but I find it fascinating how patterns in sound can evoke emotions. What kind of music do you enjoy?"),
    ("What's your favorite color?", "As an AI I don't experience colors, but I find the concept of wavelengths of light fascinating. Do you have a favorite?"),
    ("Can you tell me a joke?", "Sure! Why do programmers prefer dark mode? Because light attracts bugs! 😄"),
    ("Tell me something interesting.", "Here's something: the human brain contains about 86 billion neurons, each connected to thousands of others — making it the most complex structure known."),
    ("I'm bored.", "I can help with that! We could discuss a topic you're curious about, explore a science question, or I can explain how AI models work. What sounds good?"),
    ("I'm learning programming.", "That's great! Programming is a powerful skill. Are you learning a specific language? I can help explain concepts, debug logic, or suggest learning resources."),
    ("What should I learn first in AI?", "Start with Python, then linear algebra and calculus basics. Then study machine learning fundamentals — gradient descent, neural networks. PyTorch or TensorFlow are great frameworks to practice with."),
    ("Is AI dangerous?", "AI has both risks and benefits. The main concerns are misuse, bias in models, and long-term safety. Responsible development with human oversight is key."),
    ("Will AI replace humans?", "AI augments human capabilities more than it replaces them. Repetitive tasks may be automated, but creativity, empathy, and complex judgment remain distinctly human strengths."),
    ("How do I get started with machine learning?",
     "Start with Python and NumPy. Learn the basics of statistics and linear algebra. Then try scikit-learn for classical ML, and PyTorch or TensorFlow for deep learning. Kaggle is a great place to practice."),
    ("What programming language should I learn?",
     "Python is the most versatile choice today — it's used in AI, web, automation, and data science. JavaScript is great for web development. C is excellent for understanding how computers work."),
    ("What is a good project for learning AI?",
     "Great starter projects: image classifier with CNNs, text sentiment analysis, a simple chatbot, or training a small language model from scratch. Start small and iterate!"),
    ("How does ChatGPT work?",
     "ChatGPT is based on the GPT (Generative Pre-trained Transformer) architecture. It's trained on large text corpora with next-token prediction, then fine-tuned with reinforcement learning from human feedback (RLHF)."),
    ("What is the difference between AI and ML?",
     "AI is the broad field of making machines intelligent. ML is a subset of AI focused on learning from data. Deep learning is a subset of ML using multi-layer neural networks."),
    ("Que penses-tu de l'avenir de l'IA?",
     "L'IA va continuer à transformer de nombreux secteurs. Les défis importants incluent la sécurité, l'équité et l'alignement des valeurs. C'est un domaine passionnant avec beaucoup de travail important à faire."),
    ("Tu parles français?", "Oui! Je peux répondre en français ou en anglais selon votre préférence. Dans quelle langue souhaitez-vous discuter?"),
    ("Est-ce que tu peux m'expliquer les réseaux de neurones?",
     "Bien sûr! Un réseau de neurones est composé de couches de nœuds interconnectés. Chaque nœud applique une transformation à son entrée. L'entraînement ajuste les poids pour minimiser l'erreur sur les données."),
]

EXTENDED_TECH = [
    ("What is version control?",
     "Version control tracks changes to code over time, allowing you to revert to previous states, collaborate, and branch. Git is the most widely used version control system."),
    ("What is Git?",
     "Git is a distributed version control system. It lets developers track changes, create branches for features, and merge work. GitHub and GitLab host Git repositories online."),
    ("What is an API?",
     "An API (Application Programming Interface) defines how software components interact. It specifies what functions are available, what inputs they take, and what they return."),
    ("What is a database?",
     "A database is an organized collection of data. Relational databases (like PostgreSQL) store data in tables with rows and columns. NoSQL databases handle unstructured data."),
    ("What is SQL?",
     "SQL (Structured Query Language) is used to query and manipulate relational databases. You can SELECT rows, INSERT data, UPDATE records, and DELETE entries."),
    ("What is a REST API?",
     "A REST API uses HTTP methods (GET, POST, PUT, DELETE) to expose resources at URLs. It's stateless, meaning each request contains all the information needed."),
    ("What is Docker?",
     "Docker packages applications and their dependencies into containers — lightweight, portable units that run consistently across different environments."),
    ("What is Linux?",
     "Linux is an open-source operating system kernel. It powers servers, Android, supercomputers, and embedded systems. Popular distributions include Ubuntu, Debian, and Arch."),
    ("What is a compiler?",
     "A compiler translates high-level source code into machine code (or bytecode) that a computer can execute. GCC compiles C/C++; `javac` compiles Java."),
    ("What is memory management?",
     "Memory management controls how programs allocate and free memory. In C, you manually call malloc/free. In Python/Java, a garbage collector handles this automatically."),
    ("What is a pointer in C?",
     "A pointer in C stores the memory address of a variable. You declare it with *, dereference with *, and get an address with &. Pointers are fundamental to low-level programming."),
    ("What is the difference between stack and heap?",
     "The stack stores local variables and function call frames — it's fast but limited in size. The heap is for dynamic allocation (malloc in C) — larger but requires manual management."),
    ("What is Big O notation?",
     "Big O notation describes the time or space complexity of an algorithm as input size grows. O(1) is constant, O(n) is linear, O(n²) is quadratic, O(log n) is logarithmic."),
    ("What is a hash table?",
     "A hash table maps keys to values using a hash function. It provides average O(1) lookup, insert, and delete. Python dictionaries and C++ unordered_map use hash tables."),
    ("What is recursion?",
     "Recursion is when a function calls itself with a smaller subproblem. Every recursive solution needs a base case to stop. Factorial, Fibonacci, and tree traversal are classic examples."),
    ("What is multithreading?",
     "Multithreading runs multiple threads within one process, sharing memory. It enables parallelism on multi-core CPUs. Care must be taken to avoid race conditions and deadlocks."),
    ("What is OpenMP?",
     "OpenMP is an API for shared-memory parallel programming in C, C++, and Fortran. You add #pragma omp directives to parallelize loops across CPU cores with minimal code changes."),
    ("What is SIMD?",
     "SIMD (Single Instruction, Multiple Data) executes one instruction on multiple data elements simultaneously. CPU extensions like SSE, AVX, and AVX-512 provide SIMD instructions for vectorized computation."),
    ("What is the difference between float and double?",
     "float is 32-bit floating point (≈7 decimal digits of precision). double is 64-bit (≈15 decimal digits). Use float for GPU/ML workloads where memory and speed matter; double for numerical precision."),
    ("What is matrix multiplication?",
     "Matrix multiplication of A[M,K] and B[K,N] produces C[M,N] where C[i,j] = sum_k A[i,k]*B[k,j]. It's O(MNK) and fundamental to neural network forward passes."),
]

CONVERSATIONAL_FLOWS = [
    [
        ("Can you explain how language models work?",
         "Language models learn to predict the next token given a sequence. They're trained on large text corpora and learn grammar, facts, and reasoning patterns from the statistics of language."),
        ("How do they generate text?",
         "At each step, the model outputs a probability distribution over the vocabulary. A token is sampled from that distribution, appended to the context, and the process repeats."),
        ("What determines the creativity of the output?",
         "Temperature controls randomness. High temperature (>1) makes outputs more diverse and creative. Low temperature (<0.5) makes them more focused and deterministic. Temperature=0 is fully greedy."),
    ],
    [
        ("I want to build my own AI.",
         "Great ambition! Start by understanding the fundamentals: linear algebra, calculus, and probability. Then learn a deep learning framework like PyTorch."),
        ("What should I build first?",
         "A great first project is a simple character-level language model — like BissiMamba! It teaches you embeddings, forward passes, loss functions, and optimization in one project."),
        ("How long will it take?",
         "With consistent effort, you can build a working small language model in a few weeks. Mastering deep learning takes months to years, but every step teaches you something real."),
    ],
    [
        ("What is the difference between supervised and unsupervised learning?",
         "Supervised learning trains on labeled data (input → output pairs). Unsupervised learning finds patterns in unlabeled data, like clustering or dimensionality reduction."),
        ("Which is better?",
         "Neither is universally better — it depends on your data and goal. Supervised learning is more common for practical applications. Unsupervised is powerful when labels are expensive."),
        ("What about self-supervised learning?",
         "Self-supervised learning generates labels from the data itself — like predicting the next word. This is how language models like GPT are trained, combining the benefits of both approaches."),
    ],
    [
        ("How do neural networks learn?",
         "They learn by minimizing a loss function. The forward pass computes predictions; backpropagation computes gradients; an optimizer updates weights to reduce the loss."),
        ("What is the loss function?",
         "The loss function measures how wrong the model's predictions are. For language models, cross-entropy loss measures the difference between predicted and actual token probabilities."),
        ("How many training steps are needed?",
         "It varies widely. Small models may need thousands of steps; large models need billions. You iterate until the loss stops decreasing or you reach your compute budget."),
    ],
    [
        ("Tell me about the C programming language.",
         "C is a low-level systems programming language created in the 1970s. It gives direct control over memory and hardware, making it ideal for operating systems, embedded systems, and performance-critical code."),
        ("Is C still used today?",
         "Absolutely! Linux, Python's interpreter, most embedded systems, and high-performance libraries are written in C. It remains one of the most influential and widely-used languages."),
        ("What's the hardest part of C?",
         "Memory management. You manually allocate and free memory with malloc/free. Bugs like buffer overflows, use-after-free, and memory leaks are common pitfalls that require discipline to avoid."),
    ],
    [
        ("What makes a good conversational AI?",
         "A good conversational AI understands context across turns, gives helpful and accurate answers, admits when it doesn't know something, and communicates naturally."),
        ("How do you maintain context?",
         "By including the conversation history in the input. The model sees the full dialogue (up to its context window) and generates responses conditioned on everything said so far."),
        ("What is a context window?",
         "The context window is the maximum number of tokens the model can consider at once. Larger windows (2048, 4096, or more tokens) let the model remember longer conversations."),
    ],
]

CODING_QA = [
    ("How do I reverse a string in Python?", "In Python: `s[::-1]` reverses a string using slice notation. Or: `''.join(reversed(s))`. Both are O(n)."),
    ("How do I sort a list in Python?", "`list.sort()` sorts in-place; `sorted(list)` returns a new sorted list. Both accept a `key` argument and `reverse=True`."),
    ("How do I read a file in C?", "Use `fopen(path, \"r\")` to open, `fgets` or `fread` to read, and `fclose` to close. Always check the return value of `fopen` for NULL."),
    ("What is a segmentation fault?", "A segfault occurs when a program accesses memory it's not allowed to — like dereferencing a NULL pointer or going out of bounds. Use `gdb` or `valgrind` to debug."),
    ("How does malloc work?", "`malloc(n)` allocates n bytes on the heap and returns a pointer. Always check for NULL (allocation failure) and call `free()` when done to avoid memory leaks."),
    ("What is a Makefile?", "A Makefile automates building software. It defines targets, dependencies, and commands. `make` runs the default target; `make target` runs a specific one."),
    ("How do I compile a C program?", "`gcc -O2 -o output source.c -lm` compiles source.c to an executable. `-O2` optimizes, `-lm` links the math library."),
    ("What is a struct in C?", "A struct groups related variables of different types under one name. `struct Point { float x; float y; };` defines a 2D point type."),
    ("What is a function pointer in C?", "A function pointer holds the address of a function. `int (*fp)(int, int) = &add;` declares a pointer to a function taking two ints and returning int."),
    ("How do I handle errors in C?", "Check return values of functions. Many return -1 or NULL on error. Use `perror()` or `strerror(errno)` to print descriptive error messages."),
]


# ── Generation ────────────────────────────────────────────────────────

def write_pair(f, human: str, bot: str):
    f.write(f"Human: {human}\nBot: {bot}\n\n")

def generate_dataset(n_repeat: int = 8):
    count = 0
    with open(OUT, "w", encoding="utf-8") as f:

        all_pairs = (GREETINGS + FAREWELLS + THANKS + IDENTITY
                     + TECH_QA + SCIENCE_QA + GENERAL_QA + SMALLTALK
                     + EXTENDED_TECH + CODING_QA)

        for _ in range(n_repeat):
            random.shuffle(all_pairs)
            for h, b in all_pairs:
                write_pair(f, h, b)
                count += 1

        # Multi-turn flows
        for _ in range(n_repeat * 2):
            for flow in CONVERSATIONAL_FLOWS:
                for h, b in flow:
                    write_pair(f, h, b)
                    count += 1

        # Numerical variations
        ops = [("add", "+"), ("subtract", "-"), ("multiply", "*")]
        for _ in range(200):
            a, b_n = random.randint(1, 999), random.randint(1, 999)
            name, sym = random.choice(ops)
            if sym == "+":   res = a + b_n
            elif sym == "-": res = a - b_n
            else:            res = a * b_n
            write_pair(f,
                f"What is {a} {sym} {b_n}?",
                f"{a} {sym} {b_n} = {res}.")
            count += 1

        # Sentence completion variations
        starters = [
            ("The capital of {} is", [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                                       ("Spain","Madrid"),("Brazil","Brasília"),("Canada","Ottawa"),
                                       ("Australia","Canberra"),("Japan","Tokyo"),("China","Beijing"),
                                       ("India","New Delhi"),("Senegal","Dakar"),("Morocco","Rabat")]),
        ]
        for _ in range(3):
            for tmpl, pairs in starters:
                for country, capital in pairs:
                    write_pair(f,
                        f"What is the capital of {country}?",
                        f"The capital of {country} is {capital}.")
                    count += 1

    return count

if __name__ == "__main__":
    print("Generating synthetic conversation dataset…")
    n = generate_dataset(n_repeat=10)
    size = OUT.stat().st_size
    print(f"Done: {n} pairs, {size:,} bytes ({size//1024} KB) → {OUT}")
    print("\nWorkflow:")
    print("  make mamba_lm_train && ./mamba_lm_train            # CPU small model")
    print("  make train_large && ./train_large --small          # CUDA 7M model")
    print("  make train_large && ./train_large                  # CUDA 1B model")
