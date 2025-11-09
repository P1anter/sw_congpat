// API_BASE_URL은 main.py가 실행되는 8000번 포트를 기준으로 합니다.
const API_BASE_URL = "http://127.0.0.1:8000/api";

// 1. itemid 매핑 정보를 저장할 변수
let itemidMap = {}; // main.py로부터 로드될 예정

document.addEventListener("DOMContentLoaded", () => {
    const subjectIdInput = document.getElementById("subject-id-input");
    const searchBtn = document.getElementById("search-btn");
    const predictBtn = document.getElementById("predict-btn");
    
    // 출력 영역 요소
    const infoFields = {
        'subject-id': document.getElementById('info-subject-id'),
        'gender': document.getElementById('info-gender'),
        'age': document.getElementById('info-age'),
        'year': document.getElementById('info-year'),
        'dod': document.getElementById('info-dod'),
    };
    
    const predTimeDisplay = document.getElementById('pred-dischtime-display');
    const predictStatusOutput = document.getElementById("predict-status-output");
    const predictionDetails = document.getElementById("prediction-details");
    
    // 새 변수: 환자 검사 결과를 표시할 영역
    const testResultsArea = document.getElementById("test-results-area"); 

    // 초기화 함수
    function resetOutputs(initial = false) {
        Object.values(infoFields).forEach(el => el.textContent = '---');
        
        predTimeDisplay.textContent = initial ? '---' : '---';
        
        // 검사 결과 영역 초기화
        testResultsArea.innerHTML = initial ? '<p class="guide-text">환자 조회 시 검사 기록을 가져옵니다.</p>' : '';

        predictStatusOutput.textContent = '';
        predictionDetails.innerHTML = initial ? '<p class="guide-text">조회 후 퇴원일 예측 버튼을 눌러주세요.</p>' : '';
    }
    resetOutputs(true);

    // 입원기간 계산 함수 (Req 3 반영)
    function calculateLOS(admissionStr, dischargeStr) {
        if (!admissionStr || !dischargeStr) {
            return 'N/A';
        }
        try {
            const date_adm = new Date(admissionStr);
            const date_pred = new Date(dischargeStr);
            
            if (isNaN(date_adm.getTime()) || isNaN(date_pred.getTime())) {
                return 'N/A';
            }
            
            const diff_time = date_pred.getTime() - date_adm.getTime();
            const diff_days = (diff_time / (1000 * 3600 * 24)).toFixed(1);
            
            return `${diff_days}일`;
        } catch (e) {
            return 'N/A';
        }
    }

    // 새로운 함수: 검사 결과를 HTML 테이블로 변환하여 표시 (수정됨)
    function renderTestResults(tests) {
        if (!tests || tests.length === 0) {
            testResultsArea.innerHTML = '<p class="guide-text">검사 기록이 없습니다.</p>';
            return;
        }

        let tableHTML = "<table>";
        // 수정: '입원 ID'와 '입원일'을 제거하고, '항목 ID'를 '검사명'으로 변경
        tableHTML += "<thead><tr><th>기록 시간</th><th>검사명</th><th>결과 값</th><th>단위</th></tr></thead>";
        tableHTML += "<tbody>";

        tests.forEach(test => {
            const charttime = test.charttime ? test.charttime.split(' ')[0] : 'N/A'; // 날짜만 표시
            
            // itemid를 한국어 검사명으로 매핑 (정수 키를 문자열 키로 찾음)
            const itemidDisplay = itemidMap[String(test.itemid)] || test.itemid || 'N/A';
            
            // value가 '___' 또는 null이면 valuenum을 사용, 아니면 value 사용
            const value_display = (test.value === '___' || test.value === null) ? 
                                (test.valuenum !== null ? test.valuenum : 'N/A') : 
                                (test.value || 'N/A');
            
            tableHTML += "<tr>";
            // 수정: 입원 ID (test.hadm_id) 제거
            tableHTML += `<td>${charttime}</td>`;
            // 수정: 항목 ID 대신 매핑된 검사명 출력
            tableHTML += `<td>${itemidDisplay}</td>`; 
            tableHTML += `<td>${value_display}</td>`;
            tableHTML += `<td>${test.valueuom || 'N/A'}</td>`;
            // 수정: 입원일 (test.days_of_visit) 제거
            tableHTML += "</tr>";
        });

        tableHTML += "</tbody></table>";
        testResultsArea.innerHTML = tableHTML;
    }


    // 1. 환자 조회 및 검사 기록 로드
    searchBtn.addEventListener("click", async () => {
        const subjectId = subjectIdInput.value;
        if (!subjectId) { 
            predictStatusOutput.textContent = "환자 ID를 입력하세요."; 
            resetOutputs();
            return; 
        }

        resetOutputs();
        predictStatusOutput.textContent = "환자 정보 조회 중...";
        testResultsArea.innerHTML = '<p class="guide-text">검사 기록 조회 중...</p>';
        
        try {
            // 1. 환자 기본 정보 조회
            const patientResponse = await fetch(`${API_BASE_URL}/patient/${subjectId}`);
            const patientData = await patientResponse.json();

            if (!patientResponse.ok) { throw new Error(patientData.detail || "환자 조회에 실패했습니다."); }
            
            // 데이터 출력
            infoFields['subject-id'].textContent = patientData.subject_id || 'N/A';
            infoFields['gender'].textContent = patientData.gender || 'N/A';
            infoFields['age'].textContent = patientData.anchor_age || 'N/A';
            infoFields['year'].textContent = patientData.anchor_year || 'N/A';
            infoFields['dod'].textContent = patientData.dod || 'N/A';
            
            // 2. 환자 검사 기록 조회 (새 API 호출)
            const testsResponse = await fetch(`${API_BASE_URL}/tests/${subjectId}`);
            const testsData = await testsResponse.json();

            if (!testsResponse.ok) {
                // 검사 기록 조회는 실패해도 기본 정보는 표시 (오류 메시지)
                testResultsArea.innerHTML = `<p class="guide-text">검사 기록 조회 실패: ${testsData.detail || 'API 오류'}</p>`;
                predictStatusOutput.textContent = `✅ subject_id=${patientData.subject_id} 조회 완료. (검사 기록 로드 오류 발생)`;
            } else {
                // 수정: itemidMap 전역 변수에 매핑 정보 저장
                itemidMap = testsData.item_id_map || {}; 
                
                renderTestResults(testsData.lab_tests); // 테이블 렌더링
                predictStatusOutput.textContent = `✅ subject_id=${patientData.subject_id} 조회 및 검사 기록 로드 완료. 예측을 시작하세요.`;
            }

        } catch (error) {
            resetOutputs();
            predictStatusOutput.textContent = `❌ 오류: ${error.message}`;
        }
    });

    // 2. 퇴원일 예측
    predictBtn.addEventListener("click", async () => {
        const subjectId = subjectIdInput.value;
        if (!subjectId || infoFields['subject-id'].textContent === '---') { 
            predictStatusOutput.textContent = "먼저 환자 정보를 조회해야 합니다."; 
            return; 
        }

        predTimeDisplay.textContent = '계산 중...';
        predictStatusOutput.textContent = "퇴원일 예측 중... (모델 로딩 시간 필요)";
        predictionDetails.innerHTML = '';

        try {
            const response = await fetch(`${API_BASE_URL}/predict/${subjectId}`);
            const data = await response.json();

            if (!response.ok) { throw new Error(data.detail || "예측에 실패했습니다."); }
            
            // 백엔드(main.py)에서 간결하게 수정한 status_message를 그대로 사용
            predictStatusOutput.textContent = data.status_message; 
            
            // 예측 결과 테이블 생성
            if (data.predictions && data.predictions.length > 0) {
                const firstPrediction = data.predictions[0];
                
                // 예상 퇴원일 요약 출력
                predTimeDisplay.textContent = firstPrediction.pred_dischtime ? 
                                            firstPrediction.pred_dischtime.split(' ')[0] : 'N/A'; 
                
                
                // 상세 테이블 생성 (Req 3 반영)
                let table = "<table><thead><tr>";
                const headers = ["admittime", "dischtime_true", "pred_dischtime", "los_calculated", "error_days"]; 

                const headerMap = {
                    "admittime": "입원일", 
                    "dischtime_true": "실제 퇴원일", 
                    "pred_dischtime": "예측 퇴원일", 
                    "los_calculated": "입원기간(예측)", 
                    "error_days": "오차(일)"
                };
                headers.forEach(h => table += `<th>${headerMap[h] || h}</th>`);
                table += "</tr></thead><tbody>";

                data.predictions.forEach(row => {
                    table += "<tr>";
                    const admittime = row.admittime ? row.admittime.split(' ')[0] : 'N/A';
                    const dischtime_true = row.dischtime_true ? row.dischtime_true.split(' ')[0] : 'N/A';
                    const pred_dischtime = row.pred_dischtime ? row.pred_dischtime.split(' ')[0] : 'N/A';
                    
                    const los_calculated = calculateLOS(row.admittime, row.pred_dischtime);
                    
                    table += `<td>${admittime}</td>`;
                    table += `<td>${dischtime_true}</td>`;
                    table += `<td>${pred_dischtime}</td>`;
                    table += `<td>${los_calculated}</td>`; 
                    table += `<td>${row.error_days !== null && row.error_days !== undefined ? row.error_days : 'N/A'}</td>`;
                    table += "</tr>";
                });
                
                table += "</tbody></table>";
                predictionDetails.innerHTML = table;
            }

        } catch (error) {
            predTimeDisplay.textContent = '예측 실패';
            predictStatusOutput.textContent = `❌ 예측 오류: ${error.message}`;
        }
    });
});