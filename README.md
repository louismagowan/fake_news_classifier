<div id="top"></div>
<!--
*** Copied from https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<h3 align="center">Fake News Classification with Keras</h3>


<!-- ABOUT THE PROJECT -->
## About The Project

This project looks at building, training and evaluating neural networks to solve the [fake news classification problem](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) on Kaggle. It uses Python and Keras. 

Gated Recurrent Units (GRUs) are used, along with batch normalization and both spaCy and Keras embedding layers. 

Keras embedded models appear to outperform those with a static spaCy embedding and manage to achieve extremely high test accuracies (99%+).

The run_models.ipynb notebook contains a demo of how code from the two Python files can be used to easily implement the networks needed to solve this task.

<b>Caveat:</b> The Kaggle dataset is not ideal- there are leakages (that I've tried to remove) and issues around its provenance. However, the papers that produced the dataset are reasonably well cited (~200 citations) on Google Scholar. In any case, this project is presented as more of a fun tutorial/look at how deep learning approaches could be used to solve the problem of fake news identification, rather than any serious, real-world classifier.

Feel free to check out the Medium article on this [here](https://medium.com/@louismagowan42)
<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [Keras](https://keras.io/)
* [spaCy](https://spacy.io/)
* [Tensorflow](https://www.tensorflow.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

* LinkedIn- [Louis Magowan](https://www.linkedin.com/in/louismagowan/)
* Medium - [Louis Magowan](https://medium.com/@louismagowan42)
* Project Link: [https://github.com/louismagowan/fake_news_classifier](https://github.com/louismagowan/fake_news_classifier)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [othneildrew - README template](https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md)
* [LSE Department of Statistics](https://www.lse.ac.uk/statistics)
* [Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* [Information Leaks in Kaggle Fake News Dataset](https://www.kaggle.com/code/mosewintner/5-data-leaks-100-acc-1-word-99-6-acc?scriptVersionId=45283074)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/louismagowan/fake_news_classifier.svg?style=for-the-badge
[contributors-url]: https://github.com/louismagowan/fake_news_classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/louismagowan/fake_news_classifier.svg?style=for-the-badge
[forks-url]: https://github.com/louismagowan/fake_news_classifier/network/members
[stars-shield]: https://img.shields.io/github/stars/louismagowan/fake_news_classifier.svg?style=for-the-badge
[stars-url]: https://github.com/louismagowan/fake_news_classifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/louismagowan/fake_news_classifier.svg?style=for-the-badge
[issues-url]: https://github.com/louismagowan/fake_news_classifier/issues
[license-shield]: https://img.shields.io/github/license/louismagowan/fake_news_classifier.svg?style=for-the-badge
[license-url]: https://github.com/louismagowan/fake_news_classifier/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/louismagowan/
[product-screenshot]: images/screenshot.png
