"""
Full NLLB-200 (FLORES-200) language catalog — 200 languages supported offline via CTranslate2.
Vietnamese (`vie_Latn`) is the primary source language.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LanguageEntry:
    code: str          # NLLB-200 FLORES code, e.g. "eng_Latn"
    name_vi: str       # Vietnamese display name
    name_en: str       # English display name
    native: str        # Native script name
    tts_voice: str = ""  # Optional Edge-TTS voice identifier
    pinned: bool = False


# Complete official NLLB-200 FLORES-200 language table (200 languages)
LANGUAGES: list[LanguageEntry] = [
    # --- Pinned / Popular Languages (Priority: EN, TH, LO, ZH, JA, KO, KM, FR, DE, ES, RU) ---
    LanguageEntry("eng_Latn", "Tiếng Anh", "English", "English", "en-US-JennyNeural", True),
    LanguageEntry("tha_Thai", "Tiếng Thái", "Thai", "ไทย", "th-TH-PremwadeeNeural", True),
    LanguageEntry("lao_Laoo", "Tiếng Lào", "Lao", "ລາວ", "lo-LA-KeomanyNeural", True),
    LanguageEntry("zho_Hans", "Tiếng Trung (Giản thể)", "Chinese (Simplified)", "简体中文", "zh-CN-XiaoxiaoNeural", True),
    LanguageEntry("jpn_Jpan", "Tiếng Nhật", "Japanese", "日本語", "ja-JP-NanamiNeural", True),
    LanguageEntry("kor_Hang", "Tiếng Hàn", "Korean", "한국어", "ko-KR-SunHiNeural", True),
    LanguageEntry("khm_Khmr", "Tiếng Khmer (Campuchia)", "Khmer", "ភាសាខ្មែរ", "km-KH-SreymomNeural", True),
    LanguageEntry("fra_Latn", "Tiếng Pháp", "French", "Français", "fr-FR-DeniseNeural", True),
    LanguageEntry("deu_Latn", "Tiếng Đức", "German", "Deutsch", "de-DE-KatjaNeural", True),
    LanguageEntry("spa_Latn", "Tiếng Tây Ban Nha", "Spanish", "Español", "es-ES-ElviraNeural", True),
    LanguageEntry("rus_Cyrl", "Tiếng Nga", "Russian", "Русский", "ru-RU-SvetlanaNeural", True),
    LanguageEntry("zho_Hant", "Tiếng Trung (Phồn thể)", "Chinese (Traditional)", "繁體中文", "zh-TW-HsiaoChenNeural", True),
    LanguageEntry("ita_Latn", "Tiếng Ý", "Italian", "Italiano", "it-IT-ElsaNeural", True),
    LanguageEntry("por_Latn", "Tiếng Bồ Đào Nha", "Portuguese", "Português", "pt-BR-FranciscaNeural", True),
    LanguageEntry("ind_Latn", "Tiếng Indonesia", "Indonesian", "Bahasa Indonesia", "id-ID-GadisNeural", True),
    LanguageEntry("arb_Arab", "Tiếng Ả Rập", "Arabic (Modern Standard)", "العربية", "ar-SA-ZariyahNeural", True),
    LanguageEntry("hin_Deva", "Tiếng Hindi", "Hindi", "हिन्दी", "hi-IN-SwaraNeural", True),
    LanguageEntry("zsm_Latn", "Tiếng Malaysia", "Malay", "Bahasa Melayu", "ms-MY-YasminNeural", True),
    LanguageEntry("tgl_Latn", "Tiếng Tagalog (Philippines)", "Tagalog", "Tagalog", "fil-PH-BlessicaNeural", True),
    LanguageEntry("nld_Latn", "Tiếng Hà Lan", "Dutch", "Nederlands", "nl-NL-ColetteNeural", True),
    LanguageEntry("pol_Latn", "Tiếng Ba Lan", "Polish", "Polski", "pl-PL-ZofiaNeural", True),
    LanguageEntry("tur_Latn", "Tiếng Thổ Nhĩ Kỳ", "Turkish", "Türkçe", "tr-TR-EmelNeural", True),
    LanguageEntry("ukr_Cyrl", "Tiếng Ukraina", "Ukrainian", "Українська", "uk-UA-PolinaNeural", True),
    LanguageEntry("ces_Latn", "Tiếng Séc", "Czech", "Čeština", "cs-CZ-VlastaNeural", True),
    LanguageEntry("swe_Latn", "Tiếng Thụy Điển", "Swedish", "Svenska", "sv-SE-SofieNeural", True),
    LanguageEntry("vie_Latn", "Tiếng Việt", "Vietnamese", "Tiếng Việt", "vi-VN-HoaiMyNeural", False),

    # --- All Remaining NLLB-200 Languages (Alphabetical by code) ---
    LanguageEntry("ace_Arab", "Tiếng Aceh (Ả Rập)", "Acehnese (Arabic script)", "بهسا اچيه"),
    LanguageEntry("ace_Latn", "Tiếng Aceh (Latin)", "Acehnese (Latin script)", "Bahsa Acèh"),
    LanguageEntry("acm_Arab", "Tiếng Ả Rập Iraq", "Mesopotamian Arabic", "عربي عراقي", "ar-IQ-RanaNeural"),
    LanguageEntry("acq_Arab", "Tiếng Ả Rập Yemen", "Ta'izzi-Adeni Arabic", "عربي يمني", "ar-YE-MaryamNeural"),
    LanguageEntry("aeb_Arab", "Tiếng Ả Rập Tunisia", "Tunisian Arabic", "تونسي", "ar-TN-ReemNeural"),
    LanguageEntry("afr_Latn", "Tiếng Afrikaans", "Afrikaans", "Afrikaans", "af-ZA-AdriNeural"),
    LanguageEntry("ajp_Arab", "Tiếng Ả Rập Nam Levant", "South Levantine Arabic", "شامي جنوبي", "ar-JO-SanaNeural"),
    LanguageEntry("aka_Latn", "Tiếng Akan", "Akan", "Akan"),
    LanguageEntry("als_Latn", "Tiếng Albania (Tosk)", "Tosk Albanian", "Shqip", "sq-AL-AnilaNeural"),
    LanguageEntry("amh_Ethi", "Tiếng Amhara", "Amharic", "አማርኛ", "am-ET-MekdesNeural"),
    LanguageEntry("apc_Arab", "Tiếng Ả Rập Bắc Levant", "North Levantine Arabic", "شامي شمالي", "ar-LB-LaylaNeural"),
    LanguageEntry("ars_Arab", "Tiếng Ả Rập Najdi", "Najdi Arabic", "نجدي", "ar-SA-ZariyahNeural"),
    LanguageEntry("ary_Arab", "Tiếng Ả Rập Maroc", "Moroccan Arabic", "الدارجة", "ar-MA-MounaNeural"),
    LanguageEntry("arz_Arab", "Tiếng Ả Rập Ai Cập", "Egyptian Arabic", "مصري", "ar-EG-SalmaNeural"),
    LanguageEntry("asm_Beng", "Tiếng Assam", "Assamese", "অসমীয়া"),
    LanguageEntry("ast_Latn", "Tiếng Asturias", "Asturian", "Asturianu"),
    LanguageEntry("awa_Deva", "Tiếng Awadhi", "Awadhi", "अवधी"),
    LanguageEntry("ayr_Latn", "Tiếng Aymara", "Central Aymara", "Aymar aru"),
    LanguageEntry("azb_Arab", "Tiếng Azerbaijan Nam", "South Azerbaijani", "تۆرکجه"),
    LanguageEntry("azj_Latn", "Tiếng Azerbaijan Bắc", "North Azerbaijani", "Azərbaycanca", "az-AZ-BanuNeural"),
    LanguageEntry("bak_Cyrl", "Tiếng Bashkir", "Bashkir", "Башҡортса"),
    LanguageEntry("bam_Latn", "Tiếng Bambara", "Bambara", "Bamanankan"),
    LanguageEntry("ban_Latn", "Tiếng Bali", "Balinese", "Basa Bali"),
    LanguageEntry("bel_Cyrl", "Tiếng Belarus", "Belarusian", "Беларуская"),
    LanguageEntry("bem_Latn", "Tiếng Bemba", "Bemba", "Ichibemba"),
    LanguageEntry("ben_Beng", "Tiếng Bengal", "Bengali", "বাংলা", "bn-IN-TanishaaNeural"),
    LanguageEntry("bho_Deva", "Tiếng Bhojpuri", "Bhojpuri", "भोजपुरी"),
    LanguageEntry("bjn_Arab", "Tiếng Banjar (Ả Rập)", "Banjar (Arabic script)", "بنجر"),
    LanguageEntry("bjn_Latn", "Tiếng Banjar (Latin)", "Banjar (Latin script)", "Bahasa Banjar"),
    LanguageEntry("bod_Tibt", "Tiếng Tây Tạng", "Standard Tibetan", "བོད་སྐད"),
    LanguageEntry("bos_Latn", "Tiếng Bosnia", "Bosnian", "Bosanski", "bs-BA-VesnaNeural"),
    LanguageEntry("bug_Latn", "Tiếng Bugis", "Buginese", "Basa Ugi"),
    LanguageEntry("bul_Cyrl", "Tiếng Bulgaria", "Bulgarian", "Български", "bg-BG-KalinaNeural"),
    LanguageEntry("cat_Latn", "Tiếng Catalan", "Catalan", "Català", "ca-ES-JoanaNeural"),
    LanguageEntry("ceb_Latn", "Tiếng Cebuano", "Cebuano", "Sinugbuanong Binisaya"),
    LanguageEntry("cjk_Latn", "Tiếng Chokwe", "Chokwe", "Chokwe"),
    LanguageEntry("ckb_Arab", "Tiếng Kurd (Sorani)", "Central Kurdish", "کوردیی ناوەندی"),
    LanguageEntry("crh_Latn", "Tiếng Tatar Crimea", "Crimean Tatar", "Qırımtatarca"),
    LanguageEntry("cym_Latn", "Tiếng Wales", "Welsh", "Cymraeg", "cy-GB-NiaNeural"),
    LanguageEntry("dan_Latn", "Tiếng Đan Mạch", "Danish", "Dansk", "da-DK-ChristelNeural"),
    LanguageEntry("dik_Latn", "Tiếng Dinka", "Southwestern Dinka", "Thuɔŋjäŋ"),
    LanguageEntry("dyu_Latn", "Tiếng Dyula", "Dyula", "Julakan"),
    LanguageEntry("dzo_Tibt", "Tiếng Dzongkha (Bhutan)", "Dzongkha", "རྫོང་ཁ"),
    LanguageEntry("ell_Grek", "Tiếng Hy Lạp", "Greek", "Ελληνικά", "el-GR-AthinaNeural"),
    LanguageEntry("epo_Latn", "Tiếng Quốc Tế Ngữ", "Esperanto", "Esperanto"),
    LanguageEntry("est_Latn", "Tiếng Estonia", "Estonian", "Eesti", "et-EE-AnuNeural"),
    LanguageEntry("eus_Latn", "Tiếng Basque", "Basque", "Euskara"),
    LanguageEntry("ewe_Latn", "Tiếng Ewe", "Ewe", "Eʋegbe"),
    LanguageEntry("fao_Latn", "Tiếng Faroe", "Faroese", "Føroyskt"),
    LanguageEntry("fij_Latn", "Tiếng Fiji", "Fijian", "Na Vosa Vakaviti"),
    LanguageEntry("fin_Latn", "Tiếng Phần Lan", "Finnish", "Suomi", "fi-FI-NooraNeural"),
    LanguageEntry("fon_Latn", "Tiếng Fon", "Fon", "Fɔ̀ngbè"),
    LanguageEntry("fur_Latn", "Tiếng Friuli", "Friulian", "Furlan"),
    LanguageEntry("fuv_Latn", "Tiếng Fulfulde Nigeria", "Nigerian Fulfulde", "Fulfulde"),
    LanguageEntry("gaz_Latn", "Tiếng Oromo", "West Central Oromo", "Afaan Oromoo"),
    LanguageEntry("gla_Latn", "Tiếng Gaelic Scotland", "Scottish Gaelic", "Gàidhlig"),
    LanguageEntry("gle_Latn", "Tiếng Ireland", "Irish", "Gaeilge", "ga-IE-OrlaNeural"),
    LanguageEntry("glg_Latn", "Tiếng Galicia", "Galician", "Galego", "gl-ES-SabelaNeural"),
    LanguageEntry("grn_Latn", "Tiếng Guarani", "Guarani", "Avañe'ẽ"),
    LanguageEntry("guj_Gujr", "Tiếng Gujarat", "Gujarati", "ગુજરાતી", "gu-IN-DhwaniNeural"),
    LanguageEntry("hat_Latn", "Tiếng Haiti Creole", "Haitian Creole", "Kreyòl ayisyen"),
    LanguageEntry("hau_Latn", "Tiếng Hausa", "Hausa", "Hausa"),
    LanguageEntry("heb_Hebr", "Tiếng Do Thái (Hebrew)", "Hebrew", "עברית", "he-IL-HilaNeural"),
    LanguageEntry("hne_Deva", "Tiếng Chhattisgarhi", "Chhattisgarhi", "छत्तीसगढ़ी"),
    LanguageEntry("hrv_Latn", "Tiếng Croatia", "Croatian", "Hrvatski", "hr-HR-GabrijelaNeural"),
    LanguageEntry("hun_Latn", "Tiếng Hungary", "Hungarian", "Magyar", "hu-HU-NoemiNeural"),
    LanguageEntry("hye_Armn", "Tiếng Armenia", "Armenian", "Հայերեն"),
    LanguageEntry("ibo_Latn", "Tiếng Igbo", "Igbo", "Asụsụ Igbo"),
    LanguageEntry("ilo_Latn", "Tiếng Ilocano", "Ilocano", "Ilokano"),
    LanguageEntry("isl_Latn", "Tiếng Iceland", "Icelandic", "Íslenska", "is-IS-GudrunNeural"),
    LanguageEntry("jav_Latn", "Tiếng Java", "Javanese", "Basa Jawa", "jv-ID-SitiNeural"),
    LanguageEntry("kab_Latn", "Tiếng Kabyle", "Kabyle", "Taqbaylit"),
    LanguageEntry("kac_Latn", "Tiếng Jingpho", "Jingpho", "Jinghpaw"),
    LanguageEntry("kam_Latn", "Tiếng Kamba", "Kamba", "Kikamba"),
    LanguageEntry("kan_Knda", "Tiếng Kannada", "Kannada", "ಕನ್ನಡ", "kn-IN-SapnaNeural"),
    LanguageEntry("kas_Arab", "Tiếng Kashmir (Ả Rập)", "Kashmiri (Arabic script)", "कॉशुर"),
    LanguageEntry("kas_Deva", "Tiếng Kashmir (Devanagari)", "Kashmiri (Devanagari script)", "कॉशुर"),
    LanguageEntry("kat_Geor", "Tiếng Gruzia (Georgia)", "Georgian", "ქართული", "ka-GE-EkaNeural"),
    LanguageEntry("kaz_Cyrl", "Tiếng Kazakhstan", "Kazakh", "Қазақша", "kk-KZ-AigulNeural"),
    LanguageEntry("kbp_Latn", "Tiếng Kabiye", "Kabiyè", "Kabɩyɛ"),
    LanguageEntry("kea_Latn", "Tiếng Kabuverdianu", "Kabuverdianu", "Kabuverdianu"),
    LanguageEntry("khk_Cyrl", "Tiếng Mông Cổ", "Halh Mongolian", "Монгол", "mn-MN-YesuiNeural"),
    LanguageEntry("kik_Latn", "Tiếng Kikuyu", "Kikuyu", "Gĩkũyũ"),
    LanguageEntry("kin_Latn", "Tiếng Kinyarwanda", "Kinyarwanda", "Ikinyarwanda"),
    LanguageEntry("kir_Cyrl", "Tiếng Kyrgyzstan", "Kyrgyz", "Кыргызча"),
    LanguageEntry("kmb_Latn", "Tiếng Kimbundu", "Kimbundu", "Kimbundu"),
    LanguageEntry("kmr_Latn", "Tiếng Kurd (Kurmanji)", "Northern Kurdish", "Kurmancî"),
    LanguageEntry("knc_Arab", "Tiếng Kanuri (Ả Rập)", "Central Kanuri (Arabic script)", "Kanuri"),
    LanguageEntry("knc_Latn", "Tiếng Kanuri (Latin)", "Central Kanuri (Latin script)", "Kanuri"),
    LanguageEntry("kon_Latn", "Tiếng Kikongo", "Kikongo", "Kikongo"),
    LanguageEntry("lij_Latn", "Tiếng Liguria", "Ligurian", "Ligure"),
    LanguageEntry("lim_Latn", "Tiếng Limburg", "Limburgish", "Limburgs"),
    LanguageEntry("lin_Latn", "Tiếng Lingala", "Lingala", "Lingála"),
    LanguageEntry("lit_Latn", "Tiếng Litva", "Lithuanian", "Lietuvių", "lt-LT-OnaNeural"),
    LanguageEntry("lmo_Latn", "Tiếng Lombard", "Lombard", "Lombard"),
    LanguageEntry("ltg_Latn", "Tiếng Latgalia", "Latgalian", "Latgaļu"),
    LanguageEntry("ltz_Latn", "Tiếng Luxembourg", "Luxembourgish", "Lëtzebuergesch"),
    LanguageEntry("lua_Latn", "Tiếng Luba-Kasai", "Luba-Kasai", "Tshiluba"),
    LanguageEntry("lug_Latn", "Tiếng Ganda", "Ganda", "Luganda"),
    LanguageEntry("luo_Latn", "Tiếng Luo", "Luo", "Dholuo"),
    LanguageEntry("lus_Latn", "Tiếng Mizo", "Mizo", "Mizo ṭawng"),
    LanguageEntry("lvs_Latn", "Tiếng Latvia", "Standard Latvian", "Latviešu", "lv-LV-EveritaNeural"),
    LanguageEntry("mag_Deva", "Tiếng Magahi", "Magahi", "मगही"),
    LanguageEntry("mai_Deva", "Tiếng Maithili", "Maithili", "मैथिली"),
    LanguageEntry("mal_Mlym", "Tiếng Malayalam", "Malayalam", "മലയാളം", "ml-IN-SobhanaNeural"),
    LanguageEntry("mar_Deva", "Tiếng Marathi", "Marathi", "मराठी", "mr-IN-AarohiNeural"),
    LanguageEntry("min_Latn", "Tiếng Minangkabau (Latin)", "Minangkabau (Latin script)", "Baso Minang"),
    LanguageEntry("mkd_Cyrl", "Tiếng Macedonia", "Macedonian", "Македонски", "mk-MK-MarijaNeural"),
    LanguageEntry("mlt_Latn", "Tiếng Malta", "Maltese", "Malti", "mt-MT-GraceNeural"),
    LanguageEntry("mni_Beng", "Tiếng Meitei (Manipuri)", "Meitei (Bengali script)", "মৈতৈলোন্"),
    LanguageEntry("mos_Latn", "Tiếng Mossi", "Mossi", "Mòoré"),
    LanguageEntry("mri_Latn", "Tiếng Maori", "Maori", "Te Reo Māori"),
    LanguageEntry("mya_Mymr", "Tiếng Myanmar (Miến Điện)", "Burmese", "မြန်မာဘာသာ", "my-MM-NilarNeural"),
    LanguageEntry("nno_Latn", "Tiếng Na Uy (Nynorsk)", "Norwegian Nynorsk", "Nynorsk"),
    LanguageEntry("nob_Latn", "Tiếng Na Uy (Bokmål)", "Norwegian Bokmål", "Norsk Bokmål", "nb-NO-PernilleNeural"),
    LanguageEntry("npi_Deva", "Tiếng Nepal", "Nepali", "नेपाली", "ne-NP-HemkalaNeural"),
    LanguageEntry("nso_Latn", "Tiếng Bắc Sotho", "Northern Sotho", "Sesotho sa Leboa"),
    LanguageEntry("nus_Latn", "Tiếng Nuer", "Nuer", "Thok Naath"),
    LanguageEntry("nya_Latn", "Tiếng Chichewa", "Nyanja", "Chichewa"),
    LanguageEntry("oci_Latn", "Tiếng Occitan", "Occitan", "Occitan"),
    LanguageEntry("ory_Orya", "Tiếng Odia", "Odia", "ଓଡ଼ିଆ"),
    LanguageEntry("pag_Latn", "Tiếng Pangasinan", "Pangasinan", "Pangasinan"),
    LanguageEntry("pan_Guru", "Tiếng Punjab", "Eastern Panjabi", "ਪੰਜਾਬੀ"),
    LanguageEntry("pap_Latn", "Tiếng Papiamento", "Papiamento", "Papiamentu"),
    LanguageEntry("pbt_Arab", "Tiếng Pashto", "Southern Pashto", "پښتو", "ps-AF-LatifaNeural"),
    LanguageEntry("pes_Arab", "Tiếng Ba Tư (Farsi)", "Western Persian", "فارسی", "fa-IR-DilaraNeural"),
    LanguageEntry("plt_Latn", "Tiếng Malagasy", "Plateau Malagasy", "Malagasy"),
    LanguageEntry("prs_Arab", "Tiếng Dari", "Dari", "دری"),
    LanguageEntry("quy_Latn", "Tiếng Quechua", "Ayacucho Quechua", "Runasimi"),
    LanguageEntry("ron_Latn", "Tiếng Romania", "Romanian", "Română", "ro-RO-AlinaNeural"),
    LanguageEntry("run_Latn", "Tiếng Rundi", "Rundi", "Ikirundi"),
    LanguageEntry("sag_Latn", "Tiếng Sango", "Sango", "Sängö"),
    LanguageEntry("san_Deva", "Tiếng Phạn (Sanskrit)", "Sanskrit", "संस्कृतम्"),
    LanguageEntry("sat_Beng", "Tiếng Santali", "Santali", "সাঁওতালি"),
    LanguageEntry("scn_Latn", "Tiếng Sicilia", "Sicilian", "Sicilianu"),
    LanguageEntry("shn_Mymr", "Tiếng Shan", "Shan", "လိၵ်ႈတႆး"),
    LanguageEntry("sin_Sinh", "Tiếng Sinhala (Sri Lanka)", "Sinhala", "සිංහල", "si-LK-ThiliniNeural"),
    LanguageEntry("slk_Latn", "Tiếng Slovakia", "Slovak", "Slovenčina", "sk-SK-ViktoriaNeural"),
    LanguageEntry("slv_Latn", "Tiếng Slovenia", "Slovenian", "Slovenščina", "sl-SI-PetraNeural"),
    LanguageEntry("smo_Latn", "Tiếng Samoa", "Samoan", "Gagana Sāmoa"),
    LanguageEntry("sna_Latn", "Tiếng Shona", "Shona", "chiShona"),
    LanguageEntry("snd_Arab", "Tiếng Sindhi", "Sindhi", "سنڌي"),
    LanguageEntry("som_Latn", "Tiếng Somali", "Somali", "Soomaali", "so-SO-UbaxNeural"),
    LanguageEntry("sot_Latn", "Tiếng Nam Sotho", "Southern Sotho", "Sesotho"),
    LanguageEntry("srd_Latn", "Tiếng Sardinia", "Sardinian", "Sardu"),
    LanguageEntry("srp_Cyrl", "Tiếng Serbia", "Serbian", "Српски", "sr-RS-SophieNeural"),
    LanguageEntry("ssw_Latn", "Tiếng Swati", "Swati", "SiSwati"),
    LanguageEntry("sun_Latn", "Tiếng Sunda", "Sundanese", "Basa Sunda", "su-ID-TutiNeural"),
    LanguageEntry("swh_Latn", "Tiếng Swahili", "Swahili", "Kiswahili", "sw-KE-ZuriNeural"),
    LanguageEntry("szl_Latn", "Tiếng Silesia", "Silesian", "Ślōnski"),
    LanguageEntry("tam_Taml", "Tiếng Tamil", "Tamil", "தமிழ்", "ta-IN-PallaviNeural"),
    LanguageEntry("taq_Latn", "Tiếng Tamasheq (Latin)", "Tamasheq (Latin script)", "Tamasheq"),
    LanguageEntry("taq_Tfng", "Tiếng Tamasheq (Tifinagh)", "Tamasheq (Tifinagh script)", "ⵜⴰⵎⴰⵛⴻⵖ"),
    LanguageEntry("tat_Cyrl", "Tiếng Tatar", "Tatar", "Татарча"),
    LanguageEntry("tel_Telu", "Tiếng Telugu", "Telugu", "తెలుగు", "te-IN-ShrutiNeural"),
    LanguageEntry("tgk_Cyrl", "Tiếng Tajikistan", "Tajik", "Тоҷикӣ"),
    LanguageEntry("tir_Ethi", "Tiếng Tigrinya", "Tigrinya", "ትግርኛ"),
    LanguageEntry("tpi_Latn", "Tiếng Tok Pisin", "Tok Pisin", "Tok Pisin"),
    LanguageEntry("tsn_Latn", "Tiếng Tswana", "Tswana", "Setswana"),
    LanguageEntry("tso_Latn", "Tiếng Tsonga", "Tsonga", "Xitsonga"),
    LanguageEntry("tuk_Latn", "Tiếng Turkmenistan", "Turkmen", "Türkmençe"),
    LanguageEntry("tum_Latn", "Tiếng Tumbuka", "Tumbuka", "chiTumbuka"),
    LanguageEntry("twi_Latn", "Tiếng Twi", "Twi", "Twi"),
    LanguageEntry("tzm_Tfng", "Tiếng Tamazight", "Central Atlas Tamazight", "ⵜⴰⵎⴰⵣⵉⵖⵜ"),
    LanguageEntry("uig_Arab", "Tiếng Duy Ngô Nhĩ", "Uyghur", "ئۇيغۇرچە"),
    LanguageEntry("umb_Latn", "Tiếng Umbundu", "Umbundu", "Umbundu"),
    LanguageEntry("urd_Arab", "Tiếng Urdu", "Urdu", "اردو", "ur-PK-UzmaNeural"),
    LanguageEntry("uzn_Latn", "Tiếng Uzbekistan", "Northern Uzbek", "Oʻzbekcha", "uz-UZ-MadinaNeural"),
    LanguageEntry("vec_Latn", "Tiếng Veneto", "Venetian", "Vèneto"),
    LanguageEntry("war_Latn", "Tiếng Waray", "Waray", "Winaray"),
    LanguageEntry("wol_Latn", "Tiếng Wolof", "Wolof", "Wolof"),
    LanguageEntry("xho_Latn", "Tiếng Xhosa", "Xhosa", "isiXhosa"),
    LanguageEntry("ydd_Hebr", "Tiếng Yiddish", "Eastern Yiddish", "ייִדיש"),
    LanguageEntry("yor_Latn", "Tiếng Yoruba", "Yoruba", "Yorùbá"),
    LanguageEntry("yue_Hant", "Tiếng Quảng Đông", "Yue Chinese (Cantonese)", "粵語", "zh-HK-HiuMaanNeural"),
    LanguageEntry("zul_Latn", "Tiếng Zulu", "Zulu", "isiZulu", "zu-ZA-ThandoNeural"),
]

LANGUAGE_MAP: dict[str, LanguageEntry] = {lang.code: lang for lang in LANGUAGES}


@dataclass(frozen=True)
class SourceLanguageEntry:
    code: str        # NLLB-200 FLORES code, e.g. "vie_Latn"
    locale: str      # BCP-47 locale for OS native ASR, e.g. "vi-VN"
    name_vi: str
    name_en: str
    native: str
    offline_onnx: bool = False  # True if cross-platform ONNX streaming model is supported


SOURCE_LANGUAGES: list[SourceLanguageEntry] = [
    SourceLanguageEntry("vie_Latn", "vi-VN", "Tiếng Việt", "Vietnamese", "Tiếng Việt", True),
    SourceLanguageEntry("eng_Latn", "en-US", "Tiếng Anh", "English", "English", True),
    SourceLanguageEntry("tha_Thai", "th-TH", "Tiếng Thái", "Thai", "ไทย", True),
    SourceLanguageEntry("lao_Laoo", "lo-LA", "Tiếng Lào", "Lao", "ລາວ", True),
    SourceLanguageEntry("jpn_Jpan", "ja-JP", "Tiếng Nhật", "Japanese", "日本語", True),
    SourceLanguageEntry("zho_Hans", "zh-CN", "Tiếng Trung (Giản thể)", "Chinese (Simplified)", "简体中文", True),
    SourceLanguageEntry("zho_Hant", "zh-TW", "Tiếng Trung (Phồn thể)", "Chinese (Traditional)", "繁體中文", True),
    SourceLanguageEntry("kor_Hang", "ko-KR", "Tiếng Hàn", "Korean", "한국어", True),
    SourceLanguageEntry("khm_Khmr", "km-KH", "Tiếng Khmer (Campuchia)", "Khmer", "ភាសាខ្មែរ", True),
    SourceLanguageEntry("fra_Latn", "fr-FR", "Tiếng Pháp", "French", "Français", True),
    SourceLanguageEntry("deu_Latn", "de-DE", "Tiếng Đức", "German", "Deutsch", True),
    SourceLanguageEntry("spa_Latn", "es-ES", "Tiếng Tây Ban Nha", "Spanish", "Español", True),
    SourceLanguageEntry("rus_Cyrl", "ru-RU", "Tiếng Nga", "Russian", "Русский", True),
    SourceLanguageEntry("ita_Latn", "it-IT", "Tiếng Ý", "Italian", "Italiano", True),
    SourceLanguageEntry("por_Latn", "pt-BR", "Tiếng Bồ Đào Nha", "Portuguese", "Português", True),
    SourceLanguageEntry("ind_Latn", "id-ID", "Tiếng Indonesia", "Indonesian", "Bahasa Indonesia", True),
    SourceLanguageEntry("arb_Arab", "ar-SA", "Tiếng Ả Rập", "Arabic", "العربية", True),
    SourceLanguageEntry("hin_Deva", "hi-IN", "Tiếng Hindi", "Hindi", "हिन्दी", True),
    SourceLanguageEntry("zsm_Latn", "ms-MY", "Tiếng Malaysia", "Malay", "Bahasa Melayu", True),
    SourceLanguageEntry("tgl_Latn", "fil-PH", "Tiếng Philippines (Tagalog)", "Tagalog", "Tagalog", True),
    SourceLanguageEntry("mya_Mymr", "my-MM", "Tiếng Myanmar", "Burmese", "မြန်မာဘာသာ", True),
    SourceLanguageEntry("nld_Latn", "nl-NL", "Tiếng Hà Lan", "Dutch", "Nederlands", True),
    SourceLanguageEntry("pol_Latn", "pl-PL", "Tiếng Ba Lan", "Polish", "Polski", True),
    SourceLanguageEntry("tur_Latn", "tr-TR", "Tiếng Thổ Nhĩ Kỳ", "Turkish", "Türkçe", True),
    SourceLanguageEntry("ukr_Cyrl", "uk-UA", "Tiếng Ukraina", "Ukrainian", "Українська", True),
]

SOURCE_LANGUAGE_MAP: dict[str, SourceLanguageEntry] = {s.code: s for s in SOURCE_LANGUAGES}


@dataclass(frozen=True)
class SourceAsrProfile:
    preferred_model: str       # Fast-Lane model_id ("zipformer-30m-streaming", "moonshine-en", "moonshine-ja", "moonshine-ko", "zipformer-multi", or "whisper-turbo")
    preferred_model_mb: int
    fallback_model: str        # Universal safety-net model_id ("whisper-turbo")
    recommended_pack: str      # "core" | "asia" | "global"
    asr_tier: str              # "fast" | "universal" | "experimental"
    asr_note: str = ""


_DEFAULT_UNIVERSAL_PROFILE = SourceAsrProfile(
    preferred_model="whisper-turbo",
    preferred_model_mb=540,
    fallback_model="whisper-turbo",
    recommended_pack="global",
    asr_tier="universal",
)

PARAKEET_EU_LANGS: frozenset[str] = frozenset(
    {
        "fra_Latn",
        "deu_Latn",
        "spa_Latn",
        "ita_Latn",
        "por_Latn",
        "nld_Latn",
        "pol_Latn",
        "ukr_Cyrl",
        "ces_Latn",
        "swe_Latn",
        "fin_Latn",
        "dan_Latn",
        "ell_Grek",
        "hun_Latn",
        "ron_Latn",
        "slk_Latn",
        "bul_Cyrl",
        "hrv_Latn",
    }
)

SOURCE_LANG_ASR_PROFILE: dict[str, SourceAsrProfile] = {
    "vie_Latn": SourceAsrProfile("zipformer-30m-streaming", 49, "whisper-turbo", "core", "fast"),
    "eng_Latn": SourceAsrProfile("moonshine-en", 106, "whisper-turbo", "core", "fast"),
    "jpn_Jpan": SourceAsrProfile("moonshine-ja", 99, "whisper-turbo", "asia", "fast"),
    "kor_Hang": SourceAsrProfile("moonshine-ko", 47, "whisper-turbo", "asia", "fast"),
    "tha_Thai": SourceAsrProfile("zipformer-multi", 247, "whisper-turbo", "asia", "fast"),
    "zho_Hans": SourceAsrProfile("zipformer-multi", 247, "whisper-turbo", "asia", "fast"),
    "zho_Hant": SourceAsrProfile("zipformer-multi", 247, "whisper-turbo", "asia", "fast"),
    "ind_Latn": SourceAsrProfile("zipformer-multi", 247, "whisper-turbo", "asia", "fast"),
    "rus_Cyrl": SourceAsrProfile("zipformer-multi", 247, "whisper-turbo", "asia", "fast"),
    "arb_Arab": SourceAsrProfile("zipformer-multi", 247, "whisper-turbo", "asia", "fast"),
    # 25 European languages — Parakeet-TDT-0.6B-Ultra INT8 (465 MB, 0% GPU, 7x-10x faster than Whisper-Turbo)
    "fra_Latn": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "deu_Latn": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "spa_Latn": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "ita_Latn": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "por_Latn": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "nld_Latn": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "pol_Latn": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "ukr_Cyrl": SourceAsrProfile("parakeet-eu", 465, "whisper-turbo", "global", "fast"),
    "lao_Laoo": SourceAsrProfile(
        "whisper-turbo",
        540,
        "whisper-turbo",
        "global",
        "experimental",
        "Nghe trực tiếp giọng Lào Offline hiện độ chính xác còn hạn chế; chiều Việt → Lào hoạt động tốt 100% Offline.",
    ),
}


def get_source_asr_profile(code: str) -> SourceAsrProfile:
    return SOURCE_LANG_ASR_PROFILE.get(code, _DEFAULT_UNIVERSAL_PROFILE)

# NLLB-200 code -> Whisper ISO-639-1 language code (99-language Whisper Large-v3-Turbo)
NLLB_TO_WHISPER_LANG: dict[str, str] = {
    "vie_Latn": "vi",
    "eng_Latn": "en",
    "tha_Thai": "th",
    "lao_Laoo": "lo",
    "jpn_Jpan": "ja",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "yue_Hant": "yue",
    "kor_Hang": "ko",
    "khm_Khmr": "km",
    "mya_Mymr": "my",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "rus_Cyrl": "ru",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "ind_Latn": "id",
    "zsm_Latn": "ms",
    "tgl_Latn": "tl",
    "arb_Arab": "ar",
    "hin_Deva": "hi",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
    "ces_Latn": "cs",
    "swe_Latn": "sv",
    "fin_Latn": "fi",
    "dan_Latn": "da",
    "nob_Latn": "no",
    "ell_Grek": "el",
    "heb_Hebr": "he",
    "hun_Latn": "hu",
    "ron_Latn": "ro",
    "slk_Latn": "sk",
    "bul_Cyrl": "bg",
    "hrv_Latn": "hr",
    "ben_Beng": "bn",
    "tam_Taml": "ta",
    "urd_Arab": "ur",
    "pes_Arab": "fa",
    "swh_Latn": "sw",
    "khk_Cyrl": "mn",
    "kaz_Cyrl": "kk",
}


def get_whisper_lang_code(code: str) -> str:
    return NLLB_TO_WHISPER_LANG.get(code, "")


def get_language(code: str) -> LanguageEntry | None:
    return LANGUAGE_MAP.get(code)


def get_source_language(code: str) -> SourceLanguageEntry | None:
    return SOURCE_LANGUAGE_MAP.get(code)


def list_languages() -> list[dict]:
    return [asdict(lang) for lang in LANGUAGES]


def list_source_languages() -> list[dict]:
    out: list[dict] = []
    for s in SOURCE_LANGUAGES:
        d = asdict(s)
        prof = get_source_asr_profile(s.code)
        d.update(
            {
                "preferred_model": prof.preferred_model,
                "preferred_model_mb": prof.preferred_model_mb,
                "fallback_model": prof.fallback_model,
                "recommended_pack": prof.recommended_pack,
                "asr_tier": prof.asr_tier,
                "asr_note": prof.asr_note,
            }
        )
        out.append(d)
    return out


# Official 38 languages/dialects supported by Tencent Hy-MT2-1.8B (Apache 2.0)
HYMT2_LANG_NAMES: dict[str, str] = {
    "vie_Latn": "Vietnamese",
    "eng_Latn": "English",
    "tha_Thai": "Thai",
    "khm_Khmr": "Khmer",
    "mya_Mymr": "Burmese",
    "zho_Hans": "Chinese",
    "zho_Hant": "Traditional Chinese",
    "yue_Hant": "Cantonese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "arb_Arab": "Arabic",
    "ita_Latn": "Italian",
    "tur_Latn": "Turkish",
    "zsm_Latn": "Malay",
    "ind_Latn": "Indonesian",
    "tgl_Latn": "Filipino",
    "hin_Deva": "Hindi",
    "pol_Latn": "Polish",
    "ces_Latn": "Czech",
    "nld_Latn": "Dutch",
    "pes_Arab": "Persian",
    "guj_Gujr": "Gujarati",
    "urd_Arab": "Urdu",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "heb_Hebr": "Hebrew",
    "ben_Beng": "Bengali",
    "tam_Taml": "Tamil",
    "ukr_Cyrl": "Ukrainian",
    "bod_Tibt": "Tibetan",
    "kaz_Cyrl": "Kazakh",
    "khk_Cyrl": "Mongolian",
    "uig_Arab": "Uyghur",
}


def get_hymt2_lang_name(code: str) -> str | None:
    return HYMT2_LANG_NAMES.get(code)


