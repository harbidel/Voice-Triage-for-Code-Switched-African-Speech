# Ethics & Inclusion Note

**Consent.** Audio is processed only after explicit spoken or written consent,
captured in the language the patient speaks. Consent for care is not consent for
data retention; these are asked separately.

**Data minimisation.** The intake record stores a pseudonymous queue reference,
never a name. Audio is processed in-memory and not persisted by default. Any
audio contributed to the optional benchmark set is de-identified and re-consented.

**Safety over fluency.** The system escalates on uncertainty and never de-escalates
on model confidence. Every tier is marked as requiring clinician confirmation. The
tool supports a triage nurse; it does not replace one.

**The inequity we are measuring.** ASR trained predominantly on Western English
degrades on African-accented and code-switched speech. In a clinic that
degradation is not a quality issue, it is a safety issue: the patients least well
served by the model are those least able to switch into the model's preferred
language. Our benchmark reports under-triage rate precisely because that is where
this inequity becomes clinical harm. A model that performs well on WER while
dropping negations and clinical entities is a model that will fail these patients
quietly.

**Known gaps.** The red-flag lexicon lacks native-speaker clinical validation. The
language-identification heuristic is lexicon-based. Neither is deployment-ready,
and we state that rather than shipping confidence we have not earned.
