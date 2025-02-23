import os
import pickle
import re

from bidi.algorithm import get_display
from langchain.schema import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import TokenTextSplitter
from nltk.tokenize import word_tokenize

from chroma import save_to_chroma_db
from config import TOKENIZED_STORAGE_PATH

text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50, add_start_index=True)

HEBREW_STOP_WORDS = set("""
כלשהו אותה קרוב האם לפיהן תמורת מבין שלי עת קרי כלומר לאור אתה שלה אצלנו נגד רוב בינו ַזֶּה מתחילת לפניכם מדי בזכות
לפיכם אחרי לעומת עמכן הגם כ לה הן מציד וְ עליך ְהַ אצל מעלי to מרבית שמא מסביב ממנו אף יותר בשבילכם לא כיוון למען
עליהן בהתחשב את פר אליו למשל אצלן בהתבסס תוך יו כול דנן לפיך מתוך וַ מעבר עמם עוד מטעם מתוכן בשבילן אלא עצמה בשבילם
אליכן לבין לאחרי מקרב זו אלה יחד איזה בעוד לפני בנוגע מה ברם עליהם כן על-פי טרם כולה אילו לקראתו ביניהם עבר מיטב בקרבת
לבד בשבילכן לפיכן and משהו למן עקב שום כמוני כִּי הואיל a אלמלא הּ בעקבות ביניכן בתור כמוהן בגלל כיאלו עלינו עצמן
אִם מעל לקראתה כאילו של אך לאחר ודאי יען מכם מהן לפיהם א דרכ לפינו מספיק כמו כמוהם כַּאֲשֶר לפניהן אולם ב זוהי כלפי
מחמת אצלכם לתוך The בלעדי דרך לפניה אצלכן הַ ל כעבור אנוכי הייתה עצמך מכן עמהן עליכם לבל בעבור לקראת באם גבי בשבילו
אנחנו אצלה בלא אני עימ כך לפניהם במקומ בשבילך לרגל כש הָ ני פרט רק לפניך ביניכם בִ כמונו למרות אותו זולת מירב You
לנוכח אם למעט שלכם בינן עצמו עַד אפילו מפאת כלשהי שלכן בידי שלהן באמצעות ככל מאשר כפי עלי אחד שכן דוגמת אתכן ו מכיוון
בגין אצלו אגב אליך אליה אותן לולא תחת נכון אותנו מבלי ערכ בינינו כמוך עם במקום במשך אשר כל הרבה בשביל מתחת "ע""פ"
מבחינת וֹ סביב מעין אֹת כמוהו אצלך אצלי אוּלָם כמה גם כמוכן ללא תחתי הודות נוכח במו ביני כלל ש לרבות מאת אֶל כיון כולן
אותי לו מן חרף בו מביני יש בקרב ביד לקראתנו שלהם כמוכם אנו מתחתי עמו Your עמך מלבד עצמם עמכם בשבילה לשם כי לפנות בשל
the לפניכן בעד פי כמוה ממ אלינו משום לקראתי פחות כולם היו אחר עמ לציד אליהם עבור ממנ עליו הללו בינך הוא טרום ביותר
לפי כולו כדי עמה היא כנגד שאר נו בשבילי בינם עובר אבל ה בתוך עמנו לקראתך לפנינו אי כגון מש מישהו לציד ממני שלנו היות
בפני משל זה מי קודם of בטרם הלא ממנה עמן מנת כלשהן באשר בעקבותי לעניין שלו כמות בין מידי לכבוד עליכן בגדר לקראתכם לפיו
בניגוד לגבי החל מספר עמי אלי בה אליבא עמהם מאז או ממך ביניהן מפני על אותם לעבר לאורך עלמנת אליהם לכדי הרי הם אלו
לפיה לקראתם מחציתכם בלי בינה יתר בשבילנו שם שני היה מעט מאחורי לקראתן מהם הלה לפניו עצמי אצלם מול מהו אליכם מאחר
לקראתכן כלשהם רובן מצד ליד אותך עליה ידי זהו אתכם שלך מתוכ לידי ךָ אֶת "ע""ש" זאת עד כאשר אל עצמנו
""".split())

def clean_hebrew_text(text):
    cleaned_text = remove_punctuation(text)
    cleaned_text = remove_english_characters(cleaned_text)
    cleaned_text = remove_extras(cleaned_text)
    cleaned_text = remove_stop_words(cleaned_text)

    return get_display(cleaned_text)

def remove_punctuation(text):
    return re.sub(r'[^\u0590-\u05FF\uFB1D-\uFB4F0-9\s]', '', text)

def remove_english_characters(text):
    return re.sub(r'[a-zA-Z]', '', text)

def remove_extras(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_stop_words(text):
    words = text.split()
    return ' '.join([word for word in words if word not in HEBREW_STOP_WORDS])

def remove_redundant_spaces(text):
    pattern = re.compile(r'(\b[\u0590-\u05FF]+)\s([\u0590-\u05FF])\b')

    fixed_text = pattern.sub(r'\1\2', text)

    return fixed_text

def embedd_pdfs_and_save(PDF_FOLDER):
    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing {filename}")
            documents.append(get_pdf_file_content(pdf_path, filename))

    print('Loaded all files and starting splitting process')
    splits = text_splitter.split_documents(documents)

    save_to_chroma_db(splits)


def get_pdf_file_content(pdf_path, filename, should_remove_redundant_spaces = True):
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    full_pdf = ""

    for doc in documents:
        full_pdf += get_display(doc.page_content) + ' '

    if should_remove_redundant_spaces:
        full_pdf = remove_redundant_spaces(full_pdf)

    metadata = {
        "source": filename
    }

    return Document(page_content=full_pdf, metadata=metadata)

def tokenize_and_store(PDF_FOLDER, storage_path=TOKENIZED_STORAGE_PATH, clean_text=True):
    all_tokenized_docs = {}

    name, ext = os.path.splitext(storage_path)
    insterted = "_clean" if clean_text else "_not_clean"
    storage_path = f"{name}{insterted}{ext}"

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Tokenizing {filename}")

            pdf_content = get_pdf_file_content(pdf_path=pdf_path, filename=filename, should_remove_redundant_spaces=clean_text)
            document = pdf_content.page_content

            if clean_text:
                document = clean_hebrew_text(document)
            tokenized_texts = word_tokenize(document)

            all_tokenized_docs[filename] = tokenized_texts

    if(os.path.exists(storage_path)):
        os.remove(storage_path)

    with open(storage_path, "wb") as f:
        pickle.dump(all_tokenized_docs, f)

    print(f"Tokenized documents saved at {storage_path}")


def load_tokenized_documents(storage_path=TOKENIZED_STORAGE_PATH, clean_text= True):
    name, ext = os.path.splitext(storage_path)
    insterted = "_clean" if clean_text else "_not_clean"
    storage_path = f"{name}{insterted}{ext}"

    if os.path.exists(storage_path):
        with open(storage_path, "rb") as f:
            all_tokenized_docs = pickle.load(f)
        print(f"Loaded precomputed tokenized data from {storage_path}")

        return all_tokenized_docs

    print("No precomputed data found, please run tokenize_and_store() first.")
    return []
