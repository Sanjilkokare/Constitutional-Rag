import unittest

import legal_index as li


class ArticleBoundaryCleanupTests(unittest.TestCase):
    def test_segment_articles_splits_21_and_21a(self):
        text = """
(Part III.—Fundamental Rights)
21. Protection of life and personal liberty.—No person shall be
deprived of his life or personal liberty except according to procedure
established by law.
2[21A. Right to education.—The State shall provide free and
compulsory education to all children of the age of six to fourteen years.]
22. Protection against arrest and detention in certain cases.—(1) No
person who is arrested shall be detained in custody without being informed.
"""
        segments = li.segment_articles(text, 42)
        article_map = {seg["article"]: seg["text"] for seg in segments if seg.get("article")}

        self.assertIn("21", article_map)
        self.assertIn("21A", article_map)
        self.assertIn("22", article_map)
        self.assertNotIn("21A. Right to education", article_map["21"])
        self.assertIn("21A. Right to education", article_map["21A"])

    def test_segment_articles_splits_32a_and_33(self):
        text = """
32. Remedies for enforcement of rights conferred by this Part.—(1)
The right to move the Supreme Court by appropriate proceedings is guaranteed.
132A. [Constitutional validity of State laws not to be considered in
proceedings under article 32.].—Omitted by the Constitution (Forty-third
Amendment) Act, 1977.
2[33. Power of Parliament to modify the rights conferred by this Part
in their application to Forces, etc.—Parliament may, by law, determine.]
"""
        segments = li.segment_articles(text, 50)
        articles = [seg["article"] for seg in segments if seg.get("article")]

        self.assertIn("32", articles)
        self.assertIn("32A", articles)
        self.assertIn("33", articles)

    def test_segment_articles_splits_34_from_33_when_heading_wraps(self):
        text = """
(Part III.â€”Fundamental Rights)
34. Restriction on rights conferred by this Part while martial law is
in force in any area.â€”Notwithstanding anything in the foregoing provisions
of this Part, Parliament may by law indemnify any person in the service of the
Union or of a State.
35. Legislation to give effect to the provisions of this Part.â€”
Notwithstanding anything in this Constitution, Parliament shall have power.
"""
        segments = li.segment_articles(text, 51)
        article_map = {seg["article"]: seg["text"] for seg in segments if seg.get("article")}

        self.assertIn("34", article_map)
        self.assertIn("35", article_map)
        self.assertIn("34. Restriction on rights conferred by this Part while martial law is", article_map["34"])
        self.assertIn("in force in any area.", article_map["34"])
        self.assertNotIn("35. Legislation", article_map["34"])

    def test_segment_articles_splits_225_and_226_without_bogus_article_two(self):
        text = """
225. Jurisdiction of existing High Courts.—Subject to the provisions
of this Constitution, the jurisdiction of any existing High Court shall continue.
2[226. Power of High Courts to issue certain writs.—(1)
Notwithstanding anything in article 32, every High Court shall have power
to issue directions, orders or writs.]
"""
        segments = li.segment_articles(text, 135)
        articles = [seg["article"] for seg in segments if seg.get("article")]

        self.assertIn("225", articles)
        self.assertIn("226", articles)
        self.assertNotIn("2", articles)

    def test_segment_articles_keeps_227_without_bogus_article_two(self):
        text = """
notwithstanding that the seat of such Government or authority or the residence
of such person is not within those territories.
3[226A. Constitutional validity of Central laws not to be considered in
proceedings under article 226.].—Omitted by the Constitution (Forty-third
Amendment) Act, 1977.
227. Power of superintendence over all courts by the High Court.—
Every High Court shall have superintendence over all courts.
2. Cl. (7) renumbered as cl. (4) by the Constitution (Forty-fourth Amendment) Act, 1978.
"""
        segments = li.segment_articles(text, 136)
        articles = [seg["article"] for seg in segments if seg.get("article")]

        self.assertIn("227", articles)
        self.assertIn("226A", articles)
        self.assertNotIn("2", articles)

    def test_segment_articles_splits_359a_and_360_without_bogus_article_two(self):
        text = """
(3) Every order made under clause (1) shall be laid before each House of Parliament.
3359A. [Application of this Part to the State of Punjab.].—Omitted by
the Constitution (Sixty-third Amendment) Act, 1989.
360. Provisions as to financial emergency.—(1) If the President is
satisfied that a situation has arisen whereby the financial stability or credit of
India is threatened, he may by a Proclamation make a declaration to that effect.
2. Added by the Constitution (Forty-second Amendment) Act, 1976.
"""
        segments = li.segment_articles(text, 248)
        articles = [seg["article"] for seg in segments if seg.get("article")]

        self.assertIn("359A", articles)
        self.assertIn("360", articles)
        self.assertNotIn("2", articles)

    def test_segment_articles_detects_368_not_3(self):
        text = """
PART XX
AMENDMENT OF THE CONSTITUTION
368.
1[Power of Parliament to amend the Constitution and
procedure therefor].— 2[(1) Notwithstanding anything in this Constitution,
Parliament may amend this Constitution.]
3. Art. 368 re-numbered as cl. (2) thereof by s. 3, ibid.
"""
        segments = li.segment_articles(text, 259)
        articles = [seg["article"] for seg in segments if seg.get("article")]

        self.assertIn("368", articles)
        self.assertNotIn("3", articles)

    def test_schedule_continuation_does_not_pollute_normal_page(self):
        text = """
PREAMBLE
WE, THE PEOPLE OF INDIA, having solemnly resolved to constitute India into a
SOVEREIGN SOCIALIST SECULAR DEMOCRATIC REPUBLIC.
"""
        segments = li.segment_schedule_entries(
            text,
            32,
            carry_schedule="XII",
            carry_list="III",
            carry_entry="",
        )
        self.assertEqual(segments, [])

    def test_footnoted_first_schedule_is_detected(self):
        text = """
1[FIRST SCHEDULE
[Articles 1 and 4]
I. THE STATES
Name
Territories
1. Andhra Pradesh
"""
        segments = li.segment_schedule_entries(text, 284)
        schedule_ids = [seg["schedule_id"] for seg in segments if seg.get("schedule_id")]

        self.assertIn("I", schedule_ids)

    def test_contents_page_does_not_create_structural_segments(self):
        text = """
Contents
ARTICLES
SEVENTH SCHEDULE—
List I — Union List.
List II— State List.
List III— Concurrent List.
EIGHTH SCHEDULE— Languages.
"""
        self.assertEqual(li.segment_schedule_entries(text, 31), [])

    def test_list_entry_split_preserves_two_digit_numbers(self):
        text = """
SEVENTH SCHEDULE
(Article 246)
List I—Union List
9. Preventive detention for reasons connected with Defence.
10. Foreign affairs; all matters which bring the Union into relation with any foreign country.
11. Diplomatic, consular and trade representation.
"""
        segments = li.segment_schedule_entries(text, 341)
        entry_ids = [seg["entry_id"] for seg in segments if seg["type"] == "entry"]

        self.assertEqual(entry_ids, ["9", "10", "11"])

    def test_schedule_continuation_header_does_not_become_entry(self):
        text = """
(Seventh Schedule)
15. War and peace.
16. Foreign jurisdiction.
"""
        segments = li.segment_schedule_entries(
            text,
            342,
            carry_schedule="VII",
            carry_list="I",
            carry_entry="14",
        )
        entry_ids = [seg["entry_id"] for seg in segments if seg["type"] == "entry"]

        self.assertEqual(entry_ids, ["15", "16"])

    def test_editorial_note_does_not_inherit_entry_id(self):
        text = """
(Seventh Schedule)
1. The words and letters "specified in Part A or Part B of the First Schedule"
omitted by the Constitution (Seventh Amendment) Act, 1956.
31. Posts and telegraphs; telephones, wireless, broadcasting and other like
forms of communication.
"""
        segments = li.segment_schedule_entries(
            text,
            343,
            carry_schedule="VII",
            carry_list="I",
            carry_entry="30",
        )

        note_segments = [
            seg for seg in segments
            if "The words and letters" in seg["text"]
        ]
        self.assertEqual(len(note_segments), 1)
        self.assertEqual(note_segments[0]["type"], "text_block")
        self.assertEqual(note_segments[0]["entry_id"], "")

        entry_ids = [seg["entry_id"] for seg in segments if seg["type"] == "entry"]
        self.assertEqual(entry_ids, ["31"])

    def test_inline_schedule_reference_is_not_treated_as_schedule_heading(self):
        text = "First Schedule and includes any other territory comprised within the territory of India."
        self.assertEqual(li.segment_schedule_entries(text, 258), [])


if __name__ == "__main__":
    unittest.main()
