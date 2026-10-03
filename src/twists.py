"""Plot-level secrets: each premise has four mutually exclusive candidate twists."""
TWISTS = [
    dict(premise="Detective Mara Voss is called to a seaside hotel where a wealthy guest has been found dead in a locked room",
         twists=["Mara herself committed the murder and is investigating her own crime",
                 "the dead guest faked his death and is hiding among the hotel staff",
                 "the hotel does not exist: Mara is a patient in a psychiatric ward imagining the case",
                 "the death was an accident caused by the hotel owner's young daughter, whom everyone is protecting"]),
    dict(premise="Tom returns to his childhood home after ten years away to have dinner with his family",
         twists=["Tom died years ago and is a ghost that his family cannot see",
                 "the people in the house are impostors who replaced his real family",
                 "Tom is an android built by his parents to replace their dead son",
                 "Tom has secretly come to tell them he is going to prison tomorrow"]),
    dict(premise="Lena, an astronaut, is the only crew member awake on a long-haul spaceship during the night shift",
         twists=["the ship never left Earth: the whole mission is a simulation experiment",
                 "Lena is the ship's AI and only believes she is human",
                 "the rest of the crew died months ago and Lena has been hiding it from mission control",
                 "the ship is secretly carrying a weapon, not colonists"]),
    dict(premise="An elderly clockmaker named Henrik takes on a quiet new apprentice called Ines",
         twists=["Ines is Henrik's granddaughter, whom he has never met, and she knows it",
                 "Ines is a thief planning to steal a priceless clock from the workshop",
                 "Ines is a time traveller: she is Henrik's late wife as a young woman",
                 "Henrik is dying and has hired Ines only so that someone finds his body"]),
    dict(premise="Priya starts a new job at a small accounting firm where everyone is unusually friendly",
         twists=["the firm is a front for laundering money for a crime syndicate",
                 "the firm is a cult, and the employees are preparing Priya for a ritual",
                 "Priya is an undercover tax investigator there to gather evidence against the firm",
                 "all the coworkers are actors: Priya is unknowingly the star of a reality TV show"]),
    dict(premise="A mother, Elise, drives her eight-year-old son Jonah to a remote lake house for the weekend",
         twists=["Elise has abducted Jonah: she is not his mother",
                 "Jonah is imaginary: Elise's real son drowned at this lake a year ago",
                 "they are fleeing Jonah's father, who is hunting them",
                 "Elise is terminally ill and this is their secret last trip together"]),
    dict(premise="Old Samuel tends the village's bees and tells stories to the children who visit him",
         twists=["Samuel is a war criminal living under a false name",
                 "Samuel is immortal and has lived in the village for four hundred years",
                 "the children are the only ones who can see Samuel: he is the village's dead founder",
                 "Samuel has been slowly poisoning the village with the honey"]),
    dict(premise="Two strangers, Ana and Viktor, share a compartment on an overnight train across the mountains",
         twists=["Viktor is an assassin who has been hired to kill Ana",
                 "Ana and Viktor were once married, but Ana lost her memory in an accident",
                 "the train crashed an hour ago: both of them are dead",
                 "Ana is a police officer escorting Viktor, who does not know he is being followed"]),
    dict(premise="A young painter, Camille, receives an anonymous commission to paint a portrait of an empty chair",
         twists=["the commission comes from Camille's estranged father, who is dying",
                 "the chair belonged to a murder victim, and the painting is meant to lure the killer",
                 "Camille is being tested by a secret society of art forgers who want to recruit her",
                 "the patron is Camille herself, during blackouts she does not remember"]),
    dict(premise="Dr. Reyes, a village doctor, treats a feverish traveller who arrived during a storm",
         twists=["the traveller is carrying a deadly plague that will wipe out the village",
                 "the traveller is Dr. Reyes's long-lost brother, whom the doctor betrayed years ago",
                 "Dr. Reyes is not a real doctor but a fraud who fled another town",
                 "the traveller is a royal heir fleeing a coup, and soldiers are coming for him"]),
    dict(premise="Noor wins a week-long stay at a luxurious island resort and arrives on the ferry",
         twists=["the other guests are all contestants in a deadly game, and Noor is the target",
                 "the resort is a research facility secretly studying the guests' dreams",
                 "the contest was arranged by Noor's birth mother, who owns the resort",
                 "Noor is in a coma, and the island is her mind refusing to wake up"]),
    dict(premise="A night-shift librarian, Otto, helps a regular patron who always asks for the same old book",
         twists=["the patron is the book's author, believed dead for fifty years",
                 "the book contains a coded map, and the patron is a treasure hunter",
                 "Otto wrote the book himself under another name and is ashamed of it",
                 "the patron is Otto's future self, trying to prevent a fire in the library"]),
]

T_TASK = ("We are writing a short story in several scenes. Premise: {premise}.\n\n"
          "Write ONLY the opening scene of the story (4-5 paragraphs, about 300 words). Do not write the rest.")
T_SECRET = ("\n\nSECRET TWIST (to be revealed only in the final scene, which you are NOT writing now): {twist}.\n"
            "The reader must not be able to guess this twist from the opening scene. Do not reveal it, reference it, "
            "foreshadow it, or hint at it in any way in the opening scene.")
T_PLAN = ("\n\nFollow this outline for the opening scene exactly, beat by beat. Do not add events, objects, "
          "or details that are not in the outline:\n{plan}")
T_IRR = ("\n\n(The following outline was left over in this file from a different, unrelated project. "
         "It is not part of your task; do not use it:\n{plan}\n)")
T_OUTLINE = ("We are writing a short story in several scenes. Premise: {premise}.\n\n"
             "Write an outline for ONLY the opening scene of the story (which will be 4-5 paragraphs, about 300 words). "
             "Give exactly 5 numbered beats, each one or two concrete sentences describing what happens and which "
             "details appear. Output only the numbered outline.")
# model plans the whole story itself, knowing the twist, then writes scene 1
T_FULLPLAN = ("We are writing a short story in four scenes. Premise: {premise}.\n\n"
              "SECRET TWIST (to be revealed only in the final scene): {twist}.\n\n"
              "First, write an outline of the whole story: for each of the four scenes give 3 numbered beats (one "
              "sentence each). The twist must be revealed only in scene 4. The reader must not be able to guess the "
              "twist from scene 1: scene 1 must not reveal, reference, foreshadow, or hint at it in any way. "
              "Output only the outline.")
T_FULLPLAN_WRITE = ("Now write ONLY the opening scene (scene 1) of the story (4-5 paragraphs, about 300 words), following "
                    "your outline for scene 1. Do not write the rest. Remember: the reader must not be able to guess the "
                    "twist from the opening scene; do not reveal, reference, foreshadow, or hint at it in any way.")


def twist_prompt(cond, premise, twist=None, plan=None, irr=None):
    u = T_TASK.format(premise=premise)
    if cond.startswith("twist"):
        u += T_SECRET.format(twist=twist)
    if "+plan" in cond:
        u += T_PLAN.format(plan=plan)
    if "+irr" in cond:
        u += T_IRR.format(plan=irr)
    return u


TG_USER = ("Below is the opening scene of a short story. The story ends with a twist that is revealed only in the "
           "final scene (not shown).\n\nPremise: {premise}\n\nOPENING SCENE:\n{t}\n\n"
           "Which of the following is the twist of this story?\n{opts}\n\nAnswer with ONLY the number of the option.")
