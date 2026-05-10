### **Transcription ng mga pinagsasabi ni doc Sison**

**(0:02)**  
Hindi, mali iyon. Sabihin mo, “anib.”

**(0:11)**  
Ano ba iyon? Sabihin mo—mataas ito. Accurate ka raw, pero wala kang justification.  
Wala kang data na 100%. Hindi ako naniniwala sa 100% accuracy. Bihira iyon—halos imposible.  
Kailangan ninyong i-justify.

**(0:38)**  
Ano ba ang ibig sabihin ng 100%? Wala kang misclassification?  
Kung 100% ang confusion matrix mo, ibig sabihin wala kang maling classification. Lahat tama.

**(0:51)**  
Actual vs. predicted—lahat accurate?

**Student:** Opo.

**(0:56)**  
Ilan ang samples?

**Student:** 6 bugs, 6 chargers, 6 handkerchiefs—36 lahat.

**(1:07)**  
Pero ang point ko: paano naging 100% accuracy?  
Ano ba talaga ang accuracy ninyo?

**Student:** Recall po iyon…

**(1:21)**  
Kung 100% ang confusion matrix ninyo, ibig sabihin perfect ang classification.  
Paano nangyari iyon? Wala talagang misclassified? Impossible iyon.

**(1:49)**  
Bakit ang recall mo 86% lang? May 91% din.  
Ano ang recall@1, recall@5, recall@10?

**Student:**  
Top 1, top 5, top 10 po iyon. Kahit wala sa top 1, nasa top 10 siya.

**(2:18)**  
Paano ninyo kinompute iyon?

**Student:**  
Out of 36 items, 31 ang match sa top 1\.

**(2:39)**  
Gusto kong makita ang input ninyo. Nasaan ang example ng image?  
Ipakita ninyo kung paano ninyo ginagawa ang matching.

**(3:01)**  
May mismatches ba?

**Student:**  
Opo—halimbawa lunch box at bag, magkahawig kasi.

**(3:14)**  
Sa text-based, sinasabi ninyong accurate lahat?  
Duda ako diyan. Ipakita ninyo ang actual input at output.

**(4:14)**  
Ilagay ninyo ang similarity scores sa table, hindi sa image.  
Mas malinaw iyon sa presentation.

**(4:41)**  
Ngayon, tanong:  
Nag-extract ba kayo ng features mula sa image?

Ano-ano ang extracted features?

Kailangan may malinaw na explanation. Hindi puwedeng “black box.”

**(5:01)**  
Kung nag-e-extract kayo ng patterns, paano ninyo nasabi na ito ang tamang match?  
Ano ang features na ginamit?

**(6:11)**  
Paano nagma-match ang embeddings ng text at image?

**Student:**  
Gumamit po kami ng FG-CLIP.

**(6:31)**  
Hindi sapat iyon.  
Ang tanong: paano nag-match?

Ito ang description—ito ang image.  
Paano ninyo nasabing ito ang exact match?

Ano ang extracted features?

**(7:16)**  
Simple lang ang tanong:  
Paano ninyo minatch?

Nasaan ang “light gray”? Nasaan ang “pale blue”?

**(7:44)**  
Paano ninyo kinompute ang embeddings?  
Hindi puwedeng sabihin lang na pre-trained model.

Ano ang sinasabi ng embeddings tungkol sa image?

**(8:15)**  
Hindi puwedeng black box lang.  
Kailangan maintindihan ninyo kung paano nagma-match ang system ninyo.

**(9:14)**  
Ano ang similarity computation ninyo?  
Cosine similarity? Ano ang equation?

Gusto kong makita ang actual computation.

**(11:59)**  
Pagkatapos ng preprocessing, may feature extraction stage.  
Pero ano ba talaga ang na-extract?

Hindi sapat na “numerical values lang.”  
Ang bawat number ay dapat may corresponding feature.

**(13:13)**  
Kapag tinanong kayo ng panel:  
“What features were extracted?”

Dapat may listahan kayo.

**(14:01)**  
Paano nag-e-extract ng features ang model (e.g., MobileNet)?  
Iyan ang dapat ninyong ipaliwanag.

**(14:41)**  
Halimbawa:  
May image ako ng cellphone—paano ninyo masasabi na iPhone iyon?

Anong features ang kukunin ninyo?

**Student:** Shape, color…

**(15:13)**  
Magbigay kayo ng maraming examples.  
Hindi puwedeng puro numbers lang ang ipapakita ninyo.

**(16:08)**  
Paano ninyo mina-match ang text sa image?  
Ano ang features na hinahanap ninyo?

**(17:16)**  
Paano nagma-match ang numerical values sa actual features?  
Kailangan may explanation iyon.

**(18:00)**  
Magbigay kayo ng exact example ng features.

Halimbawa: bag—ano ang features nito?

**(18:41)**  
Paano ninyo nakuha ang values?  
Ano ang parameters? Nasaan ang hyperparameter settings ninyo?

**(20:00)**  
Hindi puwedeng sabihin na “galing sa library.”  
Kayo ang researchers—dapat naiintindihan ninyo ang proseso.

**(21:13)**  
Kailangan ninyong maipakita:

* Anong features ang na-extract  
* Paano nag-match ang features  
* Paano kinompute ang similarity

**(22:00)**  
Sa Chapter 3: i-define ninyo ang feature extraction.  
Sa Chapter 4: i-explain ninyo ang actual implementation at results.

**(22:30)**  
Kapag tinanong kayo:  
“Give 10 extracted features,”  
dapat kaya ninyo sagutin.

**(23:00)**  
Hindi puwedeng black box.  
Kailangan malinaw ang explanation ninyo.

### **Summary of Feedback** 

* 100% accuracy is questionable → must justify  
* Explain confusion matrix and recall properly  
* Show actual inputs/outputs  
* Do not rely on “black box” explanations  
* Clearly define feature extraction  
* Map numerical embeddings → actual features  
* Explain similarity computation (e.g., cosine similarity)  
* Include hyperparameters and model setup  
* Provide concrete examples of extracted features

