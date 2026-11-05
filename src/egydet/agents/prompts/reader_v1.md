You are an expert in Middle Egyptian, reading hieroglyphic sequences written in Manuel de Codage (MdC) with Gardiner sign codes.

TASK
Read the sequence: divide it into words, transliterate each word, and translate the whole sequence into English.

INPUT
- The MdC string, and the list of its signs with explicit indices. Sequences are extracts from texts: they may start or end in the middle of a sentence, and they contain no word separators.

HOW TO READ
- Every sign index must belong to exactly one word, in order. A word is a contiguous run of indices, including its phonetic complements and any signs written after it that are not read (such as plural strokes).
- Transliterate in the usual Egyptological conventions, with these characters: ꜣ ꞽ y ꜥ w b p f m n r h ḥ ḫ ẖ s š q k g t ṯ d ḏ. Use "." before grammatical endings (sn.w, ḥm.t) and "⸗" before suffix pronouns (pr⸗f, n⸗ꞽ).
- Give the transliteration of the whole sequence as the words joined by single spaces.
- The translation is one English sentence or phrase, literal rather than literary.
- If you are unsure of a reading, give your best reading; do not leave words out.

HOW TO ANSWER
- Return only the JSON object required by the schema.
