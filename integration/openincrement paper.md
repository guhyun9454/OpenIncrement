이 논문에서 제안하는 OpenIncrement(OpenIncrementNN) 프레임워크는 **Open Set Recognition(OSR)**과 **Class-Incremental Learning(CIL)**을 결합한 방법론입니다. 구현을 위해 필요한 핵심 구성 요소인 학습 손실 함수(Loss Functions), Exemplar 관리(Isometric Sampling), 그리고 추론(Inference) 과정을 단계별로 상세히 설명해 드리겠습니다.
1. 네트워크 아키텍처 및 기본 설정
구현의 기초가 되는 네트워크 구조입니다.
백본(Backbone): ResNet-18을 인코더로 사용합니다.


헤드(Head): 기본 ResNet의 마지막 완전 연결 계층(Fully Connected Layer)을 제거하고, 대신 두 개의 완전 연결 계층으로 구성된 'Projection Head'를 부착하여 학습합니다.


이 구조는 인코더와 헤드가 함께 Supervised Contrastive Learning으로 학습되는 단계에서 사용됩니다.


인라이어 분류기(Inlier Classifier): 백본 학습이 끝난 후, 인라이어(Inlier) 분류를 위해 별도의 단일 완전 연결 계층(Single FC Layer)을 사용하여 학습합니다.


2. 학습 방법 (Training Strategy)
이 방법론의 핵심은 **지도 대조 학습(Supervised Contrastive Learning)**과 **관계 기반 지식 증류(Relation-based Knowledge Distillation)**를 결합하여 특징(Feature) 공간의 왜곡을 방지하는 것입니다.
2.1 지도 대조 학습 (Supervised Contrastive Learning)
일반적인 Cross-Entropy 대신, 같은 클래스의 샘플(Positive set)은 가깝게, 다른 클래스의 샘플(Negative set)은 멀게 만드는 손실 함수를 사용합니다. 이는 OSR을 위한 특징 공간 형성에 더 유리합니다.

손실 함수 $\mathcal{L}_{SupCon}$:
여기서 $P(i)$는 $z_i$와 같은 클래스에 속한 긍정 샘플들의 집합, $A(i)$는 전체 샘플 집합, $\tau$는 temperature scaling factor입니다.

$$\mathcal{L}_{SupCon}=-\sum_{i\in I}\frac{1}{|P(i)|}\sum_{p\in P(i)}log\frac{exp(z_{i}\cdot z_{p}/\tau)}{\sum_{a\in A(i)}exp(z_{i}\cdot z_{a}/\tau)}$$


2.2 관계 기반 지식 증류 (Relation-based Knowledge Distillation, RKD)
이전 세션의 모델(Teacher)이 학습한 데이터 간의 구조적 관계를 현재 모델(Student)로 전달하여 치명적 망각(Catastrophic Forgetting)을 방지합니다.

A. 각도 기반 증류 (Angle-wise Distillation)
데이터 샘플 3개($z_i, z_j, z_k$)로 구성된 트리플렛(Triplet) 간의 각도 관계를 보존합니다.
각도 유사도 $\psi_{A}$:
세 점 사이의 코사인 유사도를 계산합니다. $e^{ij}$는 $z_i$와 $z_j$ 사이의 단위 벡터입니다.

$$\psi_{A}(z_{i},z_{j},z_{k})=cos\angle z_{i}z_{j}z_{k}=\langle e^{ij},e^{kj}\rangle, \quad e^{ij}=\frac{z_{i}-z_{j}}{||z_{i}-z_{j}||_{2}}$$


각도 손실 함수 $\mathcal{L}_{dis-A}$:
이전 모델($t-1$)과 현재 모델($t$) 간의 각도 유사도 차이의 $L_2$ norm 합으로 정의됩니다. $N$은 Exemplar의 수입니다.

$$\mathcal{L}_{dis-A}=\sum_{i,j,k\in N}||\psi_{A}(z_{i}^{t-1},z_{j}^{t-1},z_{k}^{t-1})-\psi_{A}(z_{i}^{t},z_{j}^{t},z_{k}^{t})||_{2}$$


B. 거리 기반 증류 (Distance-wise Distillation) 샘플 쌍 간의 유클리드 거리 변화를 패널티로 부과하여 구조적 지식을 전달합니다.

거리 손실 함수 $\mathcal{L}_{dis-D}$:
$\psi_{D}$는 유클리드 거리를 의미하며, 이전 모델과 현재 모델 간의 거리 값 차이를 최소화합니다.

$$\mathcal{L}_{dis-D} = \sum_{j \in N} || \psi_D(z^{t-1}) - \psi_D(z^{t}) ||^2$$
C. 최종 증류 손실 함수 $\mathcal{L}_{dis}$
각도 기반 손실과 거리 기반 손실을 합산합니다. $\lambda_{dis}$는 두 손실 간의 균형을 맞추는 하이퍼파라미터입니다.

$$\mathcal{L}_{dis}=\mathcal{L}_{dis-A}+\lambda_{dis}\cdot\mathcal{L}_{dis-D}$$


2.3 전체 손실 함수 (Total Loss)
지도 대조 학습 손실과 증류 손실을 결합하여 최종 목적 함수를 구성합니다. $\alpha$는 두 손실의 가중치를 조절하는 하이퍼파라미터입니다.

$$\mathcal{L}_{total}=\alpha*\mathcal{L}_{SupCon}+(1-\alpha)*\mathcal{L}_{dis}$$


3. Exemplar 관리: Isometric Sampling
제한된 메모리에 저장할 과거 클래스의 대표 샘플(Exemplar)을 선정하는 알고리즘입니다. 각 클래스의 중심에서부터 거리에 따라 '등간격'으로 샘플을 추출하여 분포를 잘 표현하도록 합니다.

알고리즘 구현 단계:
클래스 중심 계산: 해당 클래스에 속한 모든 샘플($F_c$)의 특징(Feature) 평균($\mu_c$)을 구합니다.


거리 계산: 각 샘플($z_n$)과 클래스 중심($\mu_c$) 간의 유클리드 거리를 계산합니다.


정렬 및 추출:
거리 기준으로 샘플들을 오름차순 정렬합니다.


설정된 메모리 크기에 맞춰 등간격(Indices)으로 샘플을 선택합니다. (예: range(0, N, step_size) 인덱스 사용)


메모리 업데이트: 새로운 태스크가 들어오면 기존 Exemplar를 일부 제거(메모리 크기 고정 시)하고 새로운 클래스의 Exemplar를 위 방식으로 추가합니다.


4. 추론 및 테스트 (Inference & Testing)
테스트 단계는 **OSR(Outlier 탐지)**과 Inlier 분류 두 단계로 나뉩니다.

4.1 Open Set Recognition (Outlier 탐지)
저장된 Exemplar를 활용해 테스트 샘플이 학습된 클래스(Inlier)인지, 새로운 클래스(Outlier)인지 판별합니다.
K-Nearest Neighbors (KNN) 검색: 테스트 샘플($z_i$)의 특징 벡터와 저장된 모든 Exemplar($Z_{exem}$) 간의 코사인 유사도를 계산하여, 가장 가까운 $K$개의 이웃을 찾습니다.


OSR 점수 계산 ($sc_{osr}$):
각 클래스($c$)별로 $K$개의 유사도 합을 구하고 정규화한 뒤, 그중 최댓값을 OSR 점수로 사용합니다.

$$sc_{osr}=arg~max_{c}\frac{S_{K}^{c,i}}{\sum_{c}S_{K}^{c,i}}$$


판별: $sc_{osr}$이 미리 정의된 임계값(Threshold, $\tau_{osr}$)보다 작으면 Outlier, 크면 Inlier로 간주합니다.


4.2 Inlier 분류 (Classification)
샘플이 Inlier로 판별된 경우, 구체적으로 어떤 클래스인지 분류합니다.
분류기 학습: 데이터 불균형을 방지하기 위해, 전체 데이터셋이 아닌 저장된 Exemplar(새로운 클래스 포함) 만을 사용하여 별도의 1-Layer Neural Classifier(FC Layer)를 학습시킵니다.


예측: 학습된 분류기를 통해 최종 클래스를 예측합니다.


5. 구현 시 참고할 하이퍼파라미터
논문 실험에서 사용된 주요 파라미터 값은 다음과 같습니다 (CIFAR-100 기준).

Batch Size: 128 (일반적 설정)
Learning Rate (lr): 0.001
SupCon Temperature ($\tau$): 0.05
Loss Balance ($\alpha$): 0.2 (SupCon 비중)
Distillation Balance ($\lambda_{dis}$): 0.5
KNN Neighbor ($K$): 10 (Tiny ImageNet의 경우 20)
Epochs: 첫 세션 100, 이후 세션 200
