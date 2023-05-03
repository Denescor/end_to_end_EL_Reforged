#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:13:57 2021

@author: carpentier
"""

unicode = {'\\u00bb', '\\u007d', '\\u00a1', '\\u0259', '\\u0641', '\\u0398', '\\u00fd', '\\u0940', '\\u00f9', '\\u02bc', '\\u00f6', '\\u00f8', '\\u0107', '\\u0648', '\\u0105', '\\u002c', '\\u6768', '\\u0160', '\\u015b', '\\u00c0', '\\u266f', '\\u0430', '\\u0141', '\\u1ea3', '\\u00df', '\\u2212', '\\u8650', '\\u012d', '\\u1e47', '\\u00c5', '\\u00ab', '\\u0226', '\\u0930', '\\u04a4', '\\u030d', '\\u0631', '\\u207f', '\\u00bf', '\\u2010', '\\u1e6e', '\\u00cd', '\\u00c6', '\\u30e8', '\\u1e63', '\\u6a5f', '\\u03c3', '\\u00d5', '\\u0644', '\\u2020', '\\u0104', '\\u010a', '\\u013c', '\\u0123', '\\u0159', '\\u1e6f', '\\u003a', '\\u06af', '\\u00fc', '\\u4e09', '\\u0028', '\\u03b2', '\\u2103', '\\u0191', '\\u03bc', '\\u00d1', '\\u207a', '\\u79d2', '\\u6536', '\\u1ed3', '\\u0329', '\\u0196', '\\u00fb', '\\u0435', '\\u01b0', '\\u007e', '\\u1e62', '\\u0181', '\\u1ea7', '\\u2011', '\\u03c9', '\\u201d', '\\u0165', '\\u0422', '\\u1e33', '\\u0144', '\\u00fa', '\\u1ed5', '\\u0632', '\\u0643', '\\u1ea1', '\\u011e', '\\u062a', '\\u00ee', '\\u00c2', '\\u016d', '\\u003d', '\\u2202', '\\u2605', '\\u0112', '\\u73cd', '\\u03a1', '\\u0182', '\\u00d2', '\\u0153', '\\u016f', '\\u00de', '\\u00a3', '\\u1e45', '\\u1ef1', '\\u4e45', '\\u06cc', '\\u1ea2', '\\u0152', '\\uff09', '\\u0219', '\\u0457', '\\u0283', '\\u1e5a', '\\u064e', '\\u0164', '\\u0116', '\\u018b', '\\u1ec3', '\\u00b4', '\\u002b', '\\u7248', '\\u0937', '\\u203a', '\\u4eba', '\\u002f', '\\u0136', '\\u01bc', '\\u017b', '\\u00a9', '\\u03a9', '\\u00f2', '\\u2026', '\\u00c7', '\\u0969', '\\u0198', '\\u011f', '\\u00e0', '\\u0126', '\\u018a', '\\u1edf', '\\u005e', '\\u03b4', '\\u0137', '\\u01f5', '\\u1e34', '\\u007b', '\\u00f3', '\\u01c0', '\\u00f1', '\\u1ef9', '\\u03c8', '\\ub8e8', '\\u0119', '\\u014f', '\\uff5e', '\\u016c', '\\u0358', '\\u529f', '\\u2606', '\\u00e4', '\\u012b', '\\u00c9', '\\u0173', '\\u092f', '\\u5957', '\\u1e49', '\\u1ec1', '\\u019f', '\\u02bb', '\\u0399', '\\u01c1', '\\u03d5', '\\u017a', '\\u1e0d', '\\u0148', '\\u01a4', '\\u00a2', '\\u011c', '\\u1e92', '\\u01b1', '\\u0443', '\\u00f4', '\\u2122', '\\u82e5', '\\u0967', '\\u1eaf', '\\u013e', '\\u1e46', '\\u03b1', '\\u884c', '\\u0328', '\\u0021', '\\u00aa', '\\u014d', '\\u002e', '\\u00cb', '\\u062f', '\\u0102', '\\u0155', '\\u00cf', '\\u0446', '\\u1e80', '\\u003b', '\\u6c38', '\\u0103', '\\u1e6c', '\\u203c', '\\u00dc', '\\u00b3', '\\u0145', '\\u0122', '\\u8fdb', '\\u015e', '\\u017c', '\\u043f', '\\u0442', '\\u0629', '\\u6176', '\\u1edd', '\\u2018', '\\u00ea', '\\u0060', '\\u0147', '\\u00e7', '\\u00dd', '\\u00d6', '\\u043a', '\\u00ec', '\\ufb01', '\\u0124', '\\u1e5f', '\\u0431', '\\u1e94', '\\u1ea8', '\\u01b3', '\\u018f', '\\u0627', '\\u1ef3', '\\u091f', '\\u03a6', '\\u674e', '\\u016b', '\\u039b', '\\u2032', '\\u002a', '\\u2033', '\\u00b7', '\\u00ce', '\\u2075', '\\u043c', '\\u2116', '\\u1e6d', '\\u00be', '\\u0171', '\\u0433', '\\u0635', '\\u0640', '\\u01a1', '\\u00a7', '\\u019d', '\\u301c', '\\u00f5', '\\u0187', '\\u1ef6', '\\u1e0c', '\\u1ec5', '\\u01b8', '\\u00f0', '\\u1e93', '\\u00eb', '\\u03bd', '\\u0108', '\\u010d', '\\u5229', '\\u2080', '\\u00c8', '\\u9ece', '\\u0917', '\\u85cf', '\\u1edb', '\\u1e25', '\\u045b', '\\u1ec9', '\\u1ecd', '\\u0633', '\\u2022', '\\u01e8', '\\u0197', '\\u2019', '\\u0421', '\\u1ecb', '\\u9910', '\\uc2a4', '\\u1e31', '\\u00c1', '\\u1ea9', '\\u1eeb', '\\u01f4', '\\u01c2', '\\u0146', '\\u0162', '\\ufb02', '\\u01ac', '\\u0025', '\\u015c', '\\u01ce', '\\u01c3', '\\u0179', '\\u01e5', '\\u58eb', '\\u745e', '\\uff08', '\\u016a', '\\u03a5', '\\u039c', '\\u00bd', '\\u0169', '\\u55f7', '\\u89d2', '\\u00e6', '\\u9752', '\\u005b', '\\u003f', '\\u041f', '\\u06a9', '\\u0027', '\\u1ebf', '\\u0646', '\\u0130', '\\u1eb1', '\\u0395', '\\u03b3', '\\u01d4', '\\u00ed', '\\u041a', '\\u0149', '\\u0143', '\\u010f', '\\u1e24', '\\u0121', '\\u1ecf', '\\u06c1', '\\u00d3', '\\u0029', '\\u02bf', '\\u010e', '\\u1e0e', '\\u01d2', '\\u00ff', '\\u00fe', '\\u03a0', '\\u1ebc', '\\u2153', '\\u00e1', '\\u00ca', '\\u012c', '\\u017d', '\\u0110', '\\u266d', '\\u0639', '\\u014e', '\\u1e43', '\\u0026', '\\u20ac', '\\u0024', '\\u011d', '\\u003e', '\\u0163', '\\u0939', '\\u221a', '\\u00e3', '\\u65f6', '\\u0118', '\\u0101', '\\u0628', '\\u221e', '\\u1ed1', '\\u0393', '\\u00c4', '\\u0161', '\\u00e9', '\\u0220', '\\u0115', '\\u002d', '\\u03c0', '\\u0177', '\\u00ba', '\\u0158', '\\u01a7', '\\u0117', '\\u043b', '\\u00d7', '\\u00e2', '\\u0175', '\\u0420', '\\u0391', '\\u00b2', '\\u014c', '\\u2013', '\\u00b9', '\\u1ef7', '\\u064a', '\\u0301', '\\u95a2', '\\u0113', '\\u013b', '\\u094d', '\\u03b5', '\\u1eef', '\\u2c6b', '\\u00da', '\\u00d8', '\\u0432', '\\u0109', '\\u00d9', '\\u00d4', '\\u011b', '\\u0303', '\\u0392', '\\u1eed', '\\u0444', '\\u026a', '\\u0218', '\\u00ef', '\\u1ed9', '\\u00b0', '\\u010c', '\\uac00', '\\u02be', '\\u2012', '\\u5baa', '\\u00e8', '\\u1ebd', '\\u30fb', '\\u0127', '\\u010b', '\\u0131', '\\u1ebb', '\\u0150', '\\u0327', '\\u0100', '\\u1ee7', '\\u1ed7', '\\u0129', '\\u00c3', '\\u003c', '\\u2260', '\\u0106', '\\u6625', '\\u0184', '\\u1eb5', '\\u4fdd', '\\u00b1', '\\u021b', '\\u014b', '\\uff0d', '\\u1e2a', '\\u00e5', '\\u017e', '\\u011a', '\\u1eab', '\\u200e', '\\u1e35', '\\u1e5b', '\\u2192', '\\u0040', '\\u1eb7', '\\u01b2', '\\u5b58', '\\u201c', '\\u015f', '\\u01e6', '\\u0111', '\\u738b', '\\u03a7', '\\u1ead', '\\u1ec7', '\\u0324', '\\u2665', '\\ub9c8', '\\u6bba', '\\u0151', '\\u2661', '\\u03ba', '\\ua784', '\\u2014', '\\u1ee9', '\\u0120', '\\u012a', '\\u7433', '\\u0134', '\\u039a', '\\u1ee3', '\\u1ea5', '\\u1ee5', '\\u0142', '\\u043e', '\\u01eb', '\\u0440', '\\u03a3', '\\u093e', '\\u00d0', '\\u092e', '\\u00b5', '\\u013d', '\\u1ecc', '\\u0394', '\\u00bc', '\\u01d0', '\\u015a', '\\u02b9', '\\u0645', '\\u043d', '\\u00cc'}


ascii_table = {'»', '}', '¡', 'ə', 'ف', 'Θ', 'ý', 'ी', 'ù', 'ʼ', 'ö', 'ø', 'ć', 'و', 'ą', ',', '杨', 'Š', 'ś', 'À', '♯', 'а', 'Ł', 'ả', 'ß', '−', '虐', 'ĭ', 'ṇ', 'Å', '«', 'Ȧ', 'र', 'Ҥ', 'ʼ', 'ر', 'ⁿ', '¿', '‐', 'Ṯ', 'Í', 'Æ', 'ヨ', 'ṣ', '機', 'σ', 'Õ', 'ل', '†', 'Ą', 'Ċ', 'ļ', 'ģ', 'ř', 'ṯ', ':', 'گ', 'ü', '三', '(', 'β', '℃', 'Ƒ', 'μ', 'Ñ', '⁺', '秒', '收', 'ồ', '̩', 'Ɩ', 'û', 'е', 'ư', '~', 'Ṣ', 'Ɓ', 'ầ', '‑', 'ω', '”', 'ť', 'Т', 'ḳ', 'ń', 'ú', 'ổ', 'ز', 'ك', 'ạ', 'Ğ', 'ت', 'î', 'Â', 'ŭ', '=', '∂', '★', 'Ē', '珍', 'Ρ', 'Ƃ', 'Ò', 'œ', 'ů', 'Þ', '£', 'ṅ', 'ự', '久', 'ی', 'Ả', 'Œ', '）', 'ș', 'ї', 'ʃ', 'Ṛ', 'َ', 'Ť', 'Ė', 'Ƌ', 'ể', '´', '+', '版', 'ष', '›', '人', '/', 'Ķ', 'Ƽ', 'Ż', '©', 'Ω', 'ò', '…', 'Ç', '३', 'Ƙ', 'ğ', 'à', 'Ħ', 'Ɗ', 'ở', '^', 'δ', 'ķ', 'ǵ', 'Ḵ', '{', 'ó', 'ǀ', 'ñ', 'ỹ', 'ψ', '루', 'ę', 'ŏ', '～', 'Ŭ', '͘', '功', '☆', 'ä', 'ī', 'É', 'ų', 'य', '套', 'ṉ', 'ề', 'Ɵ', 'ʻ', 'Ι', 'ǁ', 'ϕ', 'ź', 'ḍ', 'ň', 'Ƥ', '¢', 'Ĝ', 'Ẓ', 'Ʊ', 'у', 'ô', '™', '若', '१', 'ắ', 'ľ', 'Ṇ', 'α', '行', '̨', '!', 'ª', 'ō', '.', 'Ë', 'د', 'Ă', 'ŕ', 'Ï', 'ц', 'Ẁ', ';', '永', 'ă', 'Ṭ', '‼', 'Ü', '³', 'Ņ', 'Ģ', '进', 'Ş', 'ż', 'п', 'т', 'ة', '慶', 'ờ', '‘', 'ê', '`', 'Ň', 'ç', 'Ý', 'Ö', 'к', 'ì', 'ﬁ', 'Ĥ', 'ṟ', 'б', 'Ẕ', 'Ẩ', 'Ƴ', 'Ə', 'ا', 'ỳ', 'ट', 'Φ', '李', 'ū', 'Λ', '′', '*', '″', '·', 'Î', '⁵', 'м', '№', 'ṭ', '¾', 'ű', 'г', 'ص', 'ـ', 'ơ', '§', 'Ɲ', '〜', 'õ', 'Ƈ', 'Ỷ', 'Ḍ', 'ễ', 'Ƹ', 'ð', 'ẓ', 'ë', 'ν', 'Ĉ', 'č', '利', '₀', 'È', '黎', 'ग', '藏', 'ớ', 'ḥ', 'ћ', 'ỉ', 'ọ', 'س', '•', 'Ǩ', 'Ɨ', '’', 'С', 'ị', '餐', '스', 'ḱ', 'Á', 'ẩ', 'ừ', 'Ǵ', 'ǂ', 'ņ', 'Ţ', 'ﬂ', 'Ƭ', '%', 'Ŝ', 'ǎ', 'ǃ', 'Ź', 'ǥ', '士', '瑞', '（', 'Ū', 'Υ', 'Μ', '½', 'ũ', '嗷', '角', 'æ', '青', '[', '?', 'П', 'ک', '\'', 'ế', 'ن', 'İ', 'ằ', 'Ε', 'γ', 'ǔ', 'í', 'К', 'ŉ', 'Ń', 'ď', 'Ḥ', 'ġ', 'ỏ', 'ہ', 'Ó', ')', 'ʿ', 'Ď', 'Ḏ', 'ǒ', 'ÿ', 'þ', 'Π', 'Ẽ', '⅓', 'á', 'Ê', 'Ĭ', 'Ž', 'Đ', '♭', 'ع', 'Ŏ', 'ṃ', '&', '€', '$', 'ĝ', '>', 'ţ', 'ह', '√', 'ã', '时', 'Ę', 'ā', 'ب', '∞', 'ố', 'Γ', 'Ä', 'š', 'é', 'Ƞ', 'ĕ', '-', 'π', 'ŷ', 'º', 'Ř', 'Ƨ', 'ė', 'л', '×', 'â', 'ŵ', 'Р', 'Α', '²', 'Ō', '–', '¹', 'ỷ', 'ي', '́', '関', 'ē', 'Ļ', '्', 'ε', 'ữ', 'Ⱬ', 'Ú', 'Ø', 'в', 'ĉ', 'Ù', 'Ô', 'ě', '̃', 'Β', 'ử', 'ф', 'ɪ', 'Ș', 'ï', 'ộ', '°', 'Č', '가', 'ʾ', '‒', '宪', 'è', 'ẽ', '・', 'ħ', 'ċ', 'ı', 'ẻ', 'Ő', '̧', 'Ā', 'ủ', 'ỗ', 'ĩ', 'Ã', '<', '≠', 'Ć', '春', 'Ƅ', 'ẵ', '保', '±', 'ț', 'ŋ', '－', 'Ḫ', 'å', 'ž', 'Ě', 'ẫ', '‎', 'ḵ', 'ṛ', '→', '@', 'ặ', 'Ʋ', '存', '“', 'ş', 'Ǧ', 'đ', '王', 'Χ', 'ậ', 'ệ', '̤', '♥', '마', '殺', 'ő', '♡', 'κ', 'Ꞅ', '—', 'ứ', 'Ġ', 'Ī', '琳', 'Ĵ', 'Κ', 'ợ', 'ấ', 'ụ', 'ł', 'о', 'ǫ', 'р', 'Σ', 'ा', 'Ð', 'म', 'µ', 'Ľ', 'Ọ', 'Δ', '¼', 'ǐ', 'Ś', 'ʹ', 'م', 'н', 'Ì'}


assert len(unicode) == len(ascii_table)

unicode2ascii = dict()

for i,letter in enumerate(ascii_table):
    unicode2ascii[unicode[i]] = ascii_table[i]
unicode2ascii['\\u0022'] = '"'
unicode2ascii['\\u0023'] = '#'
unicode2ascii['\\u005c'] = '\\'
unicode2ascii['\\u00a0'] = ''