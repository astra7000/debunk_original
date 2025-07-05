# 피부 분석 AI 모델 개발 및 성능 개선 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 다양한 피부 상태를 몇개의 클래스(분류하고자 하는 얼굴 영역과 분류 목적에 따라 클래스 수가 다름)로 분류하는 이미지 분류 모델을 개발하는 것을 목표로 함. ResNet-50을 기반으로, 마지막 레이어의 Stride를 2에서 1로 변경하여 더 높은 해상도의 피처맵을 활용하는 모델을 사용함.

프로젝트의 가장 큰 기술적 과제는 **심각한 클래스 불균형(Class Imbalance)** 문제임. 특정 클래스의 데이터는 9,000개가 넘는 반면, 다른 클래스는 200개 미만으로 존재하여 모델이 소수 클래스를 제대로 학습하지 못하는 문제가 발생함.

이 문서는 초기 모델의 문제점을 진단하고, 다양한 해결책을 논의하며, 최종적으로 코드와 학술적 근거에 기반한 최적의 솔루션을 도출해나가는 과정을 상세히 기록함.

---

## 2. 핵심 기술 및 개념

- **Base Model**: `ResNet-50` (last layer stride: 2→1)
- **Key Challenge**: 심각한 클래스 불균형 데이터셋
- **Solutions Discussed**:
  - **Loss-Level Approach**:
    - `Class-Balanced Loss` (CB Loss)[1]
    - `Focal Loss`[2]
  - **Data-Level Approach**:
    - `Data Augmentation` (Oversampling)
  - **Optimization**:
    - `wandb.sweeps`를 이용한 하이퍼파라미터 튜닝

---

## 2-1. 클래스 불균형 상황에서의 평가 지표 선택

일반적인 분류 문제에서는 전체 샘플 중 정답을 맞춘 비율(Val Acc, 일반 정확도)이 주요 성능 지표로 사용됨. 그러나 본 프로젝트와 같이 클래스 불균형이 극심한 데이터셋에서는, Val Acc만을 기준으로 모델을 평가할 경우 다수 클래스(샘플 수가 많은 클래스)에 대한 예측 성능만이 반영되는 왜곡이 발생함. 예를 들어, 전체 데이터의 90%가 클래스 '2'에 속한다면, 모델이 모든 샘플을 '2'로 예측해도 Val Acc는 90%에 도달할 수 있음. 이 경우 소수 클래스에 대한 분류 성능은 전혀 반영되지 않음.

따라서, **Val Balanced Acc(균형 정확도)**와 **Val Macro F1**과 같은 지표를 우선적으로 고려해야 함. Val Balanced Acc는 각 클래스별 정확도를 동일하게 반영하여 평균을 내므로, 소수 클래스의 성능 개선이 전체 지표에 직접적으로 영향을 미침. Val Macro F1 역시 각 클래스의 F1 score(정밀도와 재현율의 조화 평균)를 산출한 뒤 평균을 내어, 소수 클래스의 예측 성능을 균등하게 평가함. 이 두 지표가 상승한다는 것은, 모델이 다수 클래스뿐 아니라 소수 클래스까지 고르게 학습하고 있음을 의미함.

실제로, Val Acc가 소폭 하락하더라도 Val Balanced Acc와 Val Macro F1이 크게 오르는 경우, 이는 모델이 기존에 거의 구별하지 못했던 소수 클래스들을 빠르게 학습하고 있다는 강력한 증거임. 따라서 본 프로젝트에서는 모델 저장 기준 및 하이퍼파라미터 탐색의 주요 목표 지표로 Val Balanced Acc와 Val Macro F1을 채택함.

---

## 3. 개발 및 트러블슈팅 로그

### Phase 1: 초기 모델 진단

#### 상황
- **모델**: ResNet-50 (stride=1), `Class-Balanced Loss` 사용
- **데이터**: **데이터 증강 미적용**. 원본의 불균형한 데이터셋으로 학습함.

#### 문제점
1.  **심각한 과적합**: 학습 정확도(Train Acc)는 90%를 초과하였으나, 검증 정확도(Val Acc)는 70%대에서 정체됨.
2.  **소수 클래스 학습 실패**: 전체 정확도(`Val Acc`)에 비해 클래스별 성능을 평균적으로 나타내는 `Val Balanced Acc` 및 `Val Macro F1` 점수가 현저히 낮았음. 이는 모델이 다수 클래스만 잘 맞추고, 소수 클래스는 거의 학습하지 못하고 있음을 의미함.

#### 초기 분석
CB Loss만으로는 극심한 데이터 불균형 문제를 해결하기에 역부족임. 근본적인 원인은 모델이 소수 클래스의 특징을 학습할 **기회 자체가 부족**한 것이므로, 데이터 증강을 통한 **오버샘플링(Oversampling)**이 시급하다고 판단함.

---

### Phase 2: 데이터 증강과 Class-Balanced Loss의 시너지 효과

#### 핵심 가설 설정
프로젝트의 가장 중요한 논의.
> "데이터 증강(오버샘플링)으로 학습 데이터셋의 클래스 분포를 균일하게 맞춘다면, 클래스 불균형을 전제로 하는 Class-Balanced Loss는 의미가 없어지는 것인가?"

#### 결론: 상호 배타적이 아닌, 강력한 시너지 관계임
두 기법은 서로 다른 단계에서 문제를 해결하기 때문에 함께 사용했을 때 가장 큰 효과를 냄.

- **데이터 증강 (Oversampling)**: **Data-Level 접근법**임.
  - 모델이 학습하는 동안 각 클래스를 만나게 되는 **빈도**를 인위적으로 균등하게 맞춤.
  - 소수 클래스의 특징을 학습할 **기회**를 양적으로 늘려주는 역할을 함.

- **Class-Balanced Loss**: **Loss-Level 접근법**임.
  - 예측이 틀렸을 때 발생하는 손실(Loss)의 **패널티 강도**를 조절함.
  - 데이터의 **본질적인 희소성(inherent rarity)**을 기반으로, 배우기 더 어려운 소수 클래스를 틀렸을 때 더 큰 패널티를 부여함. (Class-Balanced Loss[1])

---

### Phase 3: 코드 기반 증명 및 학술적 근거

위 시너지 효과가 단순한 추론이 아닌, 코드의 실제 동작 방식과 학술적 원리에 기반하고 있음을 확인했습니다.

#### 코드 실행 순서를 통한 증명
프로젝트의 `main.py`는 두 기법의 시너지를 위해 다음과 같이 매우 이상적으로 설계되어 있습니다.

1.  **1단계: 원본 불균형 정보 캡처**
    - 데이터 증강을 적용하기 **전**, 원본 학습 데이터셋의 불균형한 클래스 분포(예: `{'2': 5443, '5': 101, ...}`)를 `train_samples_per_cls` 변수에 저장합니다.
2.  **2단계: CB Loss 초기화**
    - `train_samples_per_cls` 변수가 `Model` 클래스로 전달되어 `CB_loss`를 초기화합니다. 이 시점에서 `CB_loss`는 각 클래스의 **원본 샘플 수**를 기반으로 한 고유의 보상 가중치를 내부적으로 계산하고 "기억"합니다.
3.  **3단계: 데이터셋 교체 및 학습**
    - 그 후, `train_dataset`이 `AugmentedSkinDataset`으로 교체되어 `DataLoader`는 모델에 균형 잡힌 데이터 스트림을 제공합니다.

결과적으로, 모델은 균등한 학습 기회를 얻는 동시에, 손실 함수는 원본 데이터의 희소성을 기억하고 패널티를 차등적으로 부과할 수 있습니다.

#### 학술적 근거
이러한 접근법은 다음의 논문들에서 제안된 원리와 일치합니다.

- **Class-Balanced Loss**:
  > Cui, Y., et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples". *CVPR*.[1]
  - **내용**: `beta` 값을 이용해 '유효 샘플 수'를 계산하고, 이를 기반으로 소수 클래스에 강력한 가중치를 부여하는 방법을 제안합니다. 이 논문은 또한 제안된 손실 함수가 데이터 샘플링 전략과 **상호 보완적**이라고 명시합니다.

- **Focal Loss**:
  > Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection". *ICCV*.[2]
  - **내용**: `gamma` 값을 이용해 모델이 쉽게 맞추는 샘플의 손실을 줄여, 어렵고 애매한 샘플에 집중하여 학습 효율을 높이는 방법을 제안합니다. (본 프로젝트의 CB Loss는 Focal Loss 메커니즘을 포함하고 있습니다.)

---

## 4. 최종 실행 계획 및 권장 사항

### 1. 데이터 증강 활성화
`main.py` 실행 시 `--use_augmentation` 플래그를 **반드시 추가**하여 `AugmentedSkinDataset`을 활성화함.

### 2. Class-Balanced Loss 유지
`--train_loss cb` 옵션을 그대로 사용하여 데이터 증강과의 시너지 효과를 극대화함.

### 3. 하이퍼파라미터 설정
- **`beta` (CB Loss)**: 원본 데이터의 극심한 불균형을 보상하기 위해 `0.999` 또는 `0.9999`와 같은 높은 값을 유지함.
- **`gamma` (CB Loss)**: `2.0`을 표준 시작점으로 사용하고, `wandb.sweeps`를 통해 최적 값을 탐색함.

### 4. 모델 저장 기준 변경
과적합된 모델이 아닌, 소수 클래스까지 잘 일반화된 최상의 모델을 저장하기 위해 저장 기준을 변경함.

**`main.py` 수정**
```python
# 기존 코드
# if val_acc > best_acc:
#     best_acc = val_acc
#     ...

# 수정 코드
best_metric = 0 # best_acc -> best_metric 으로 변수명 변경
if val_bal_acc > best_metric: # val_acc 대신 val_bal_acc 또는 val_macro_f1 사용
    best_metric = val_bal_acc
    print(f"*** New Best Model Found (Val Balanced Acc: {best_metric:.4f}) at Epoch {epoch} ***")
    # model.save_checkpoint 호출
```

### 5. `wandb.sweeps`를 이용한 최종 최적화
위의 변경 사항을 모두 적용한 후, `wandb.sweeps`를 실행하여 새로운 학습 환경에 맞는 최적의 하이퍼파라미터 조합(learning rate, dropout, gamma 등)을 탐색함.

**`sweep.yaml` 설정 예시:**
```yaml
method: bayes
metric:
  name: val_balanced_acc
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  gamma:
    values: [0.5, 1.0, 2.0, 3.0]
  dropout_prob:
    values: [0.2, 0.3, 0.4, 0.5]
  # ... 기타 파라미터
```

---

## 5. 코드 구조 및 전체 동작 과정 (요약)

- `dataset_double.py`: 데이터셋 로딩, 증강, 클래스별 샘플 수 분석을 담당함.
- `main.py`: 실험 파라미터 관리, 데이터 분할, 증강/불균형 보정, 학습/검증/테스트, wandb resume 지원을 담당함.
- `model.py`: ResNet50 기반 분류기, 다양한 손실 함수, 체크포인트 관리를 담당함.
- `utils.py`: FocalLoss, CB_loss, LabelSmoothing 등 손실 함수, 데이터 분석/증강 유틸을 담당함.
- `resnet_custom.py`: stride=1 등 커스텀 ResNet50 구현을 담당함.

### 전체 동작 순서
1. main.py 실행 → argparse로 실험 파라미터 입력
2. 데이터셋 로드 및 분석 (dataset_double.py)
3. 증강(transform) 적용 여부 결정
4. train/val/test 분할
5. 모델(Model) 생성 및 wandb, optimizer, scheduler 등 초기화
6. (resume 시) 체크포인트 및 wandb run id로 이어서 학습
7. 학습/검증/테스트 루프 진행, wandb에 실험 결과 기록
8. best model 및 실험 결과 저장

---

## 6. 사용법 예시

### 1. 새 실험 시작
```bash
python main.py --data_dirs ./data --epochs 50 --use_augmentation --wandb_name "exp1"
```

### 2. 이어서 학습 및 wandb 그래프 이어 그리기
1. wandb 대시보드에서 이어서 그리고 싶은 run의 id를 복사 (예: 2x3y4z5w)
2. 아래처럼 실행함
```bash
python main.py --resume checkpoints_YYYYMMDD_HHMMSS/best_model.pth --wandb_id 2x3y4z5w --wandb_resume allow
```

---

## 7. 기타 참고사항
- 데이터 증강은 transform 인자에 따라 유동적으로 적용됨
- CB Loss, Focal Loss 등 다양한 손실 함수 실험 가능함
- 실험 재현성, wandb 실험 관리, 체크포인트 resume 등 실전 연구에 최적화되어 있음

## 참고문헌

[1] Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9268-9277.

[2] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 2980-2988.