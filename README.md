
<br />
<p align="center">

  <h3 align="center">Pytorch Generalization</h3>

  <p align="center">
	Pytorch implementations of deep learning libraries. 
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About">About</a>
    </li>
    <li><a href="#model converting instructions">Model Converting Instructions</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#related libraries">Related Libraries</a></li>
  </ol>

<!-- ABOUT -->
## About 

I was really inspired by the work of Yiding Jiang and his group in the paper "Predicting the generalization gap in deep networks with margin distributions", and sought to replicate the work in pytorch. Along the way I noticed that many papers that use similiar methods to predict generalization lacked full pytorch implimentations, and sought to convert them. 


<!-- MODEL CONVERTING INSTRUCTIONS -->
## Model Converting Instructions

1. Clone this repo
2. Download Starting kit and Public Data from [Codalab](https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit)
3. Add path to "ingestion_program" , in line 7 of 'convert_models.py'
4. Add path to "input_data", in line 19 of 'convert_models.py'

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- References -->
## References
* [1]Y. Jiang, D. Krishnan, H. Mobahi, and S. Bengio, “Predicting the Generalization Gap in Deep Networks with Margin Distributions,” arXiv:1810.00113 [cs, stat], Jun. 2019, Accessed: Jul. 27, 2020. [Online]. Available: http://arxiv.org/abs/1810.00113.


<!-- Related Libraries -->
## Related Libraries

[generalizatio gap features tensorflow ](https://github.com/mostafaelaraby/generalization-gap-features-tensorflow)
