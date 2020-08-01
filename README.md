# Desafio Prático de PDI - Grupo ICTS

Este repositório contém a solução proposta para o Desafio Prático de PDI do Grupo ICTS.


## Considerações Iniciais

O único arquivo necessário é o *ok_test.py*, além das imagens de entrada.

Para execução do código, são necessários os seguintes módulos: numpy, cv2, imutils e math.

Os parâmetros de entrada e saída encontram-se na função *"__main__"*, ao fim do script.

Para alterar a imagem de entrada, basta especificar seu caminho na variável de entrada *input_file*.

A saída *output_status* foi especificada para estar na forma (True,) ou (False,).


## Detalhes do código

Para solucionar o desafio, foi planejada a seguinte sequência de passos:

* Determinar pontos chave na imagem para encontrar algumas propriedades, como inclinação e escala.
* Encontrar a região relacionada ao *OK*, utilizando essas propriedades
* Utilizar métodos para aprimorar essa região (rotação e redimensionamento)
* Realizar algum método para encontrar o *OK* nessa região

Para tal foram utilizados os procedimentos:

* Leitura e conversão da imagem em HSV 
* Segmentação da cor verde
* Localização da borda superior da cor verde
* Encontro dos cantos da cor verde, da escala e do ângulo da imagem
* Determinação de uma escala padrão
* Localização dos pontos que possivelmente delimitam a palavra *OK*
* Adaptação da parte da imagem interna a esses pontos
* Criação de uma máscara de *OK*
* Convolução da imagem selecionada com a máscara
* Determinação do *status* final através do resultado da convolução


## Observações

Para o funcionamento correto do código, devem ser respeitadas algumas considerações:

* A imagem de entrada deve estar com uma inclinação máxima de 45° (melhor se for menor que 30°).

* Não pode haver grandes alterações no balanço de cores, ou presença de ruído, ou presença de outras cores verdes na imagem.

* A borda superior da cor verde deve estar bem apresentada.

* Deve haver um *OK* um pouco acima da faixa de cor verde, voltado um pouco para a esquerda (considerando que a imagem esteja de cabeça para baixo).

* O *OK* deve ser visivelmente mais claro que o fundo cinza atrás dele.
