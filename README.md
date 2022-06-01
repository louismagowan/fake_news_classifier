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



<h3 align="center">15 Years of Women's Tennis Data: Algorithmic Analysis in Mostly Base Python</h3>


<!-- ABOUT THE PROJECT -->
## About The Project

In honor of Emma Raducanu's historical achievements in 2021, I look at the results of womens' tennis matches over the period 2007-2021. Your objectives are to parse the data, reconstruct tournament brackets, identify the top players, and implement algorithms to provide an alternative rankings for the players. This was project was completed as part of my [MSc Applied Social Data Science degree](https://www.lse.ac.uk/study-at-lse/Graduate/degree-programmes-2022/MSc-Applied-Social-Data-Science) at LSE.

Only fundamental Python data types are used (lists, tuples, dictionaries, numpy.ndarray, etc.) to complete this analysis. Advanced data querying packages and data analysis packages were avoided, in order to further my understanding of fundamental programming concepts.

The repository contains fifteen .csv files with match results, one file for each year. Each file contains the following variables:

Tournament – the name of the tournament that the match was part of.
Start date – the date when the tournament starts.
End date – the date when the tournament ends.
Best of – 3 means that first player to win 2 sets wins match (all WTA matches are best of 3 sets).
Player 1, Player 2 – names of the players in the match.
Rank 1, Rank 2 – WTA ranks of Player 1 and Player 2 before the start of the tournament. Not all players will have a ranking.
Set 1-3 – result for each set played where the score is shown as: number of games won by Player 1 - number of games won by Player 2. The player that wins the most games in a set wins that set.
Comment
Completed means match was played.
Player retired means that the named player withdrew and the other player won by default.

3 algorithms are used to create rankings (Winners Don't Lose, Winners Win and Winners Beat Winners)- the details for which can be found in the Jupyter Notebook.


<p align="right">(<a href="#top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

LinkedIn- [Louis Magowan](https://www.linkedin.com/in/louismagowan/)

Project Link: [https://github.com/louismagowan/tennis_algorithms](https://github.com/louismagowan/tennis_algorithms)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [othneildrew - README template](https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md)
* [LSE Applied Social Data Science](https://www.lse.ac.uk/study-at-lse/Graduate/degree-programmes-2022/MSc-Applied-Social-Data-Science)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/louismagowan/tennis_algorithms.svg?style=for-the-badge
[contributors-url]: https://github.com/louismagowan/tennis_algorithms/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/louismagowan/tennis_algorithms.svg?style=for-the-badge
[forks-url]: https://github.com/louismagowan/tennis_algorithms/network/members
[stars-shield]: https://img.shields.io/github/stars/louismagowan/tennis_algorithms.svg?style=for-the-badge
[stars-url]: https://github.com/louismagowan/tennis_algorithms/stargazers
[issues-shield]: https://img.shields.io/github/issues/louismagowan/tennis_algorithms.svg?style=for-the-badge
[issues-url]: https://github.com/louismagowan/tennis_algorithms/issues
[license-shield]: https://img.shields.io/github/license/louismagowan/tennis_algorithms.svg?style=for-the-badge
[license-url]: https://github.com/louismagowan/tennis_algorithms/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/louismagowan/
[product-screenshot]: images/screenshot.png