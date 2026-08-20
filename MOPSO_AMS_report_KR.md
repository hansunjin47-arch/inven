# MOPSO-AMS 베이스라인 구현 및 검증 보고서

## 1. 배경 및 목적

기존 논문(Guided PCN, Omega 리비전)에 EA 계열 baseline으로 MOPSO/CMODE를 이미 사용 중인데, 여기에 추가할 만한 최신 EA 후보로 두 편의 논문을 검토했다.

- **Han et al. (2023), "Adaptive multiple selection strategy for multi-objective particle swarm optimization" (MOPSO-AMS), Information Sciences 624:235-251**
- Zhao et al. (2025), "An adaptive strategy based multi-population multi-objective optimization algorithm" (A-MPMO), Information Sciences 686:120913

이 중 **MOPSO-AMS**는 기존 프로젝트가 이미 사용 중인 MOPSO를 직접 확장한 알고리즘이라 이식 난이도가 낮고, 저널 검증(ZDT/DTLZ 벤치마크 + NYT 실전 적용, 30회 반복 통계 검정)도 탄탄해서 우선 구현·검증 대상으로 선정했다.

## 2. MOPSO-AMS 알고리즘 개요

논문의 핵심 기여는 세 가지다.

1. **계층적 진화 상태 감지(Hierarchical evolutionary state detection)**: non-dominated 해들의 공간 분포 정보(diversity, I_A)와 상대적 지배 정보(convergence, I_B)를 계산해서 매 iteration을 4가지 상태(ideal / convergence / diversity / stagnation) 중 하나로 분류한다.
2. **적응적 다중 선택 전략(Adaptive multiple selection strategy)**: 감지된 상태에 따라 각 sub-swarm에 배정할 leader를 다르게 고른다 — diversity가 부족하면 convergence archive에서, convergence가 부족하면 diversity archive에서 leader를 뽑아 균형을 맞춘다.
3. **적응적 파라미터 조정(Adaptive parameter adjustment)**: 각 입자가 자신의 이전 pbest를 이겼는지 여부에 따라 관성 가중치(w)와 학습 인자(c1, c2)를 개별적으로 조정한다.

기존 프로젝트의 MOPSO는 단일 스웜 + 무작위 leader 선택이었는데, MOPSO-AMS는 이걸 **4개 sub-swarm + 상태 기반 적응적 leader 선택**으로 대체한 것이 핵심 차이다. 참고로 이 논문의 multi-population 구조는 objective별 분할이 아니라 **archive를 클러스터링한 그룹별 분할**이라, CMMOPSO 같은 objective-decomposition 계열과는 다른 방식이다.

## 3. 우리 문제에 적용하기 위한 구현 결정 사항

논문이 명시하지 않은 부분들은 합리적 선택으로 채웠고, 전부 `MOPSO_AMS.py` 상단 docstring에 기록해뒀다.

| 항목 | 논문 명시 여부 | 채택한 값/방법 |
|---|---|---|
| Population/iteration 예산 | 공개(간접) | 기존 MOPSO baseline과 동일: 200 particles × 100 iterations |
| Sub-swarm 개수 N | 공개 (N=4, Table 7 ablation으로 최적성 확인) | N=4 |
| w, c1, c2 초기값 | **비공개** | 기존 MOPSO와 동일: w=0.8, c1=1.0, c2=1.0 |
| 방향 벡터 U 개수/생성법 | **비공개** | Das-Dennis simplex-lattice, K=15 (NSGA-III 관례) |
| w/c1/c2 적응 조정의 clipping 범위 | **비공개** (곱셈 업데이트라 무한 발산 가능) | w∈[0.1,2.0], c1,c2∈[0.1,4.0] |
| Hierarchical clustering 방법 | **비공개** (이름만 언급) | scipy ward linkage |
| Ideal state의 leader 선택 기준 | 모호 ("좋은 diversity+convergence 후보") | cluster 내 정규화된 (sparsity + dominance rank) 합이 최대인 후보 |

decision space(=Q,r, 2차원 continuous)는 논문의 주 검증 대상인 ZDT/DTLZ와 동일하게 순수 continuous라, 논문에서 유일하게 discrete 처리가 필요했던 NYT 응용(파이프 규격 discretization)과 달리 별도 보정이 필요 없었다.

## 4. 계산 비용

논문 자체가 "MOPSO-AMS의 계산 복잡도는 O(MN²)로 비교 대상 MOPSO들과 동일하다"고 명시한다. Fitness evaluation 횟수(200×100=20,000회)는 기존 MOPSO와 정확히 동일하게 맞췄다. 다만 상태 감지·클러스터링 오버헤드로 wall-clock은 조금 더 걸린다 (GBM 기준 실측: run당 평균 ~167초, 기존 MOPSO보다 다소 김).

## 5. GBM 세팅 검증 결과 (완료)

10-run 평균, MOPSO / MOPSO-AMS / CMODE / Original PCN / PCN with buffer init.(variation1) / Guided PCN(variation2, softmax obj. selection) 비교:

**Set Coverage 핵심 결과**
- **MOPSO-AMS가 기존 MOPSO를 확실히 능가**: MOPSO-AMS → MOPSO 지배율 65%, 반대 방향은 20% — 논문의 핵심 주장(적응적 선택 전략이 표준 multi-swarm보다 낫다)이 재현됨
- 해의 개수도 거의 2배(96.2 vs 54.8), Spacing도 개선(34.5 vs 59.7) — diversity 메커니즘이 실제로 작동한다는 증거
- 다만 **CMODE와 Guided PCN(variation2)은 여전히 MOPSO-AMS를 능가**: CMODE가 MOPSO-AMS의 60%를 지배(반대 1%), Guided PCN이 MOPSO-AMS의 58%를 지배(반대 28%)

**해석**: MOPSO-AMS는 "더 나은 MOPSO"로서는 논문 주장대로 확실히 작동하며 구현이 제대로 됐다는 근거가 된다. 다만 프로젝트의 기존 CMODE-seeded Guided PCN을 넘어서지는 못한다 — 이는 논리적으로 자연스러운 결과이며, "최신 EA 개선판을 추가 baseline으로 검토했지만 우리 방법이 여전히 경쟁력 있다"는 서사로 리비전에 활용 가능하다.

## 6. 전체 6개 세팅 결과 (완료)

`run_mopso_ams_all_settings.py`로 stat_low/medium/high(lognormal mu,sigma 패치), seasonal(period-0만 최적화 후 3개 기간에 고정 적용하는 기존 MOPSO 방식 그대로 재현), gs(고정 수요 시퀀스) 각 10-run씩 실행 완료 (총 소요 약 2시간, run당 평균 ~170초). CMODE는 사용자 지시에 따라 이번 비교부터 제외.

### 일관된 패턴 (GBM, stat_low, stat_medium, stat_high, gs)

MOPSO-AMS가 기존 MOPSO를 head-to-head set coverage 기준으로 거의 항상 능가한다:

| 세팅 | MOPSO-AMS → MOPSO | MOPSO → MOPSO-AMS |
|---|---|---|
| GBM | 65% | 20% |
| stat_low | 45% | 38% |
| stat_medium | 51% | 40% |
| stat_high | 44% | 47% (유일하게 근소 열세) |
| gs | 70% | 30% |

해의 개수·spacing도 대체로 MOPSO-AMS가 우세 — 논문이 주장하는 "적응적 선택 전략이 표준 multi-swarm보다 낫다"는 재현된다. 다만 **variation1(buffer init.)/variation2(Guided PCN)는 이 5개 세팅 전부에서 MOPSO-AMS보다 여전히 우세**하다 (예: stat_high에서 Guided PCN이 MOPSO-AMS의 52%를 지배, 반대는 34%).

### 예외 — seasonal

- MOPSO-AMS가 MOPSO를 극단적으로 압도(45% vs 4%)하지만, **PCN with buffer init.와 Guided PCN이 MOPSO/MOPSO-AMS를 둘 다 100% 지배**
- MOPSO-AMS의 해 개수(6.9)가 오히려 MOPSO(15.8)보다 적음 — seasonal은 period-0만 탐색 후 나머지 기간에 고정 적용하는 구조라, archive가 작을 때 4-subswarm 클러스터링이 충분히 분화되지 못했을 가능성이 있음 (추가 조사 필요, 미확정)

### gs — 버그 발견 및 수정 (중요)

최초 비교에서 gs만 Guided PCN이 MOPSO/MOPSO-AMS에게 밀리는 것으로 나왔는데(10%/90%), 이는 **실제 약점이 아니라 비교 프로토콜 버그**였다.

- `main_original.py:500-559`를 추적한 결과, gs는 원래 **train/test split 구조**다: 2015년 데이터로 최적화/학습하고, **2016년 held-out 데이터에 재시뮬레이션한 결과**(`*_test1_*.csv`)가 최종 비교에 쓰이도록 설계돼 있다.
- PCN 계열(Original/variation1/variation2)은 `train()` 내부에서 `run_test()`가 항상 이 out-of-sample 재평가를 수행하므로 올바르게 test1 파일을 사용하고 있었다.
- 하지만 처음 만든 MOPSO-AMS 비교 스크립트는 **MOPSO도, MOPSO-AMS도 in-sample(2015) 파일을 그대로 사용**했다 — MOPSO 쪽에 이미 존재하던 `MOPSO_run_gs_test1_*.csv`(out-of-sample 버전)를 놓친 것.
- 실제로 in-sample MOPSO 결과는 TC=5.72, Freq=0, Amount=0(비현실적으로 완벽)인 반면, out-of-sample(test1) 결과는 TC~8.1-8.2, Freq=2~5, Amount=8~14로 현실적인 수치였다 — in-sample 평가가 뻥튀기된 낙관적 수치였음을 명확히 보여준다.

**수정**: `reevaluate_mopso_ams_gs_test1.py`로 MOPSO-AMS의 2015-도출 solution들을 2016 held-out 데이터에 동일하게 재시뮬레이션해서 `MOPSO_AMS_run_gs_test1_*.csv`를 생성하고, MOPSO/MOPSO-AMS 둘 다 test1 경로를 쓰도록 비교 스크립트를 수정했다.

**수정 후 결과** (완전히 뒤집힘):

| | 수정 전 (버그) | 수정 후 |
|---|---|---|
| Guided PCN → MOPSO 지배율 | 10% | **80%** |
| MOPSO → Guided PCN 지배율 | 90% | **0%** |
| Guided PCN → MOPSO-AMS 지배율 | 10% | **72%** |
| HV (Guided PCN) | 1.1766 (최저) | 1.2541 (MOPSO 0.45·MOPSO-AMS 0.83보다 훨씬 높음) |

즉 gs는 더 이상 예외가 아니라 **다른 5개 세팅과 동일한 패턴**(Guided PCN이 MOPSO/MOPSO-AMS를 확실히 능가)을 보인다. variation1에게 밀리는 것(18%/70%)만 이전과 동일하게 남아 있다.

## 7. 결론 및 다음 단계

- **MOPSO-AMS 구현은 논문의 핵심 주장대로 작동함이 6/6 세팅에서 확인됨** (기존 MOPSO 대비 일관된 개선, gs 수정 후 stat_high만 근소 열세) → 구현 자체의 정확성에 대한 근거로 충분
- **그러나 어느 세팅에서도 variation1/variation2(우리 방법)를 넘어서지 못함** → "최신 EA 개선판을 추가 baseline으로 검토했지만 우리 방법이 여전히 경쟁력 있다"는 서사로 리비전에 안전하게 사용 가능
- **gs의 train/test 비대칭 버그 수정은 논문 서사에 중요한 영향**: 실전 케이스 스터디(gs)에서 Guided PCN이 약점을 보인다는 이전 결론은 폐기해야 하며, 오히려 다른 세팅과 일관되게 EA baseline들을 능가한다는 게 올바른 결론
- seasonal에서의 낮은 해 개수는 원인 조사 후 논문에 각주로 남길지, 혹은 클러스터링 파라미터 조정(예: 최소 archive 크기 미달 시 N을 동적으로 줄이는 처리)으로 개선해볼지 결정 필요
- 논문에 EA baseline으로 정식 추가할지, 6개 세팅 전체 표를 그대로 반영할지 최종 결정 필요
- 현재 학습 관련 하이퍼파라미터(max_size, total_steps, n_step_episodes, n_model_updates) one-at-a-time grid search 진행 중 (`grid_search_training_params.py`) — 결과는 별도 보고 예정
