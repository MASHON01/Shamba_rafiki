# Farm Pal: a farming friend that works offline

I grew up around smallholder farming in Kenya, so I have watched the same scene play out many times. A farmer notices strange spots on their maize or bean leaves, cannot say what it is, and has no easy way to find out. The nearest extension officer is stretched across hundreds of farms. The good guidance sits inside KALRO and Infonet PDFs that a farmer with a basic phone will never open. Signal is patchy and data is expensive. By the time an answer arrives, part of the harvest is already gone.

**Farm Pal** (locally, *Shamba Rafiki*, "farm friend" in Kiswahili) is my answer to that. It is an offline farming advisor that runs on an ordinary 8 GB laptop, the kind of shared kiosk a cooperative, agro-dealer, or extension office can actually afford. A farmer asks a question in English or Kiswahili, or holds up a photo of an affected leaf, and gets practical advice drawn from real Kenyan reference material, with the sources shown. No internet is needed at the moment of asking.

## What inspired me

Two things bothered me about the usual "AI for agriculture" pitch. The first is that most of it assumes a smartphone, a data bundle, and a signal, which is not the reality for many of the farmers who need help most. The second is that a confident wrong answer in farming has a real cost: the wrong pesticide, the wrong spacing, a lost season. I wanted something that runs where the farmer is, offline, and that would rather admit it does not know than invent a recommendation.

## How I built it

Farm Pal is a retrieval-first system. A local set of KALRO, AFA, KAMIS, and Infonet-Biovision documents is processed offline into text chunks, embedded with a multilingual MiniLM model, and stored in a plain NumPy vector store. When a farmer asks something, the app finds the most relevant chunks, builds a prompt that tells the model to answer only from that material and to cite it, and runs it through a quantized Llama 3.2 1B model served by llama.cpp. A verification step then checks the answer against the sources for unsupported claims, and either approves it, adds a caveat, or replaces it. If nothing relevant was found, it says so and points the farmer to an extension officer.

The computer vision side is paired directly with the language side, not bolted on. A MobileNetV3-small classifier, trained on maize (PlantVillage) and bean (iBean) leaf images and exported to ONNX, turns a photo into a crop and disease label. That label is folded into the retrieval query and the prompt, so a photo of a rust-infected bean leaf pulls the bean-rust guidance and produces a grounded treatment answer. It reached 95.8% validation accuracy across seven classes.

For Kiswahili, I learned that a 1B model composes weak Swahili when it also has to reason over English text, so the Swahili path answers English-first and then translates the finished answer back to Kiswahili with an offline NLLB model. Everything, including translation, runs locally.

## What I learned

The biggest lesson was that **retrieval plus a small model can beat a big model** for grounded advice. Once every answer has to trace back to a real document, the smaller model's thinner built-in knowledge stops mattering as much, and you get speed and a low memory footprint for free.

I also learned to **decide by measurement, not assumption**. I benchmarked a 1B and a 3B on the same laptop and let the numbers choose. Using the official formulas,

$$S_{perf} = \min\left(\frac{\text{tokens/sec}}{15},\ 1\right)\times 100, \qquad S_{eff} = \max\left(0,\ \frac{7 - \text{peak RSS (GB)}}{7}\right)\times 100,$$

the 1B scored $S_{perf} = 100$ and $S_{eff} = 80.6$ at about 16 tokens/sec and 1.4 GB peak RAM, roughly doubling the combined contribution of the 3B while staying well under the memory ceiling. Grounding covered the quality gap.

## The challenges

The 8 GB memory limit shaped every decision. Running out of memory is disqualifying, so I treated peak RAM as a hard constraint: a small quantized model, a NumPy store instead of FAISS, ONNX instead of PyTorch at serve time, and a size-capped cache all exist to stay under the wall.

Getting the small model to behave was its own fight. It would sometimes fall into repetition loops on weak context, which I fixed with firmer repetition and frequency penalties. Retrieval needed tuning too. When I tested a bean-rust question, the system returned tomato and thrips documents, because my corpus simply had no strong bean-disease text. So I added proper bean-rust and angular-leaf-spot references and rebuilt the index, and the classifier's diagnoses finally landed on real matching guidance.

Underlying all of it was the discipline of keeping the thing reproducible: model weights and the corpus fetched or built by scripts, not committed, so a fresh machine can rebuild the whole system from a public repository.

## What's next

I kept the scope honest at maize and beans so the advice stays accurate, and the design extends by adding documents and image classes. Beyond that, I want to reach farmers who do not touch a laptop at all: a USSD and SMS channel so a basic feature phone can ask a question and get an answer back as a text, and voice input and output for low-literacy users, all with the model still running locally. Farm Pal is meant to be the friend a farmer can actually reach, and there is a lot of reach left to build.
