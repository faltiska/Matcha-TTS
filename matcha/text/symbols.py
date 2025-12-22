"""
Defines the set of symbols used in text input to the model.
"""

# Padding token used when batching sequences of different lengths
# Shorter sequences are padded with _ to match the longest sequence in the batch
# Example: ["hɛˈloʊ", "haɪ"] → ["hɛˈloʊ", "haɪ___"] (padded to length 5)
_pad = "_"

# Punctuation marks preserved during phonemization
# Helps the model learn prosody (pauses, intonation, emphasis)
# Examples:
#   . and , → pauses
#   ? and ! → rising/emphatic intonation
#   " → quoted speech patterns
# The phonemizer parameter preserve_punctuation=True keeps these in the output
_punctuation = ';:,.!?¡¿_—…-\'"«»“”()[]/ '

# IPA symbols that might appear in the list of supported languages.
# I cannot check if they are supported by eSpeak, but it probably doesn't hurt 
# to have them here, even if they will not appear in real life. 
# English, Spanish, Portuguese, French, German, Italian, Romanian, Japanese, Hebrew
ipa_symbols = (
    # Vowels
    "aeiouɑɐɒæəɘɚɛɜɝɞɨɪɔøɵɤʉʊyɶœɯʏʌᵻ"
    # Consonants
    "bβcçdðfɡɢɣhɦɧħɥjɟʝkʎlɭʟɬɫɮmɱnɳɲŋɴpɸqrɹɺɾɽɻʀʁsʂʃtʈθvʋⱱwʍxχzʐʒʑʔʕʢʡʙɕɖʜɰ"
    # Suprasegmentals
    "ˈˌːˑ‿"
    # Tone and stress markers
    "↓↑→↗↘˥˦˧˨˩"
    # Diacritics (combining and modifier)
    "ʰʱʲʷˠˤ˞ⁿˡʼʴ̩̯̃̚"
)

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(ipa_symbols)

# Special symbol ids
SPACE_ID = symbols.index(" ")
