# SYN-02 Bill Payment API

## General Requirement

|  |  |
| --- | --- |
| **Use-case Name** | Utility Bill Payment API |
| **Use-case description** | This API allows customers to pay utility bills from their bank account through the mobile channel |
| **Created by** | BA 3 |
| **Approval Status** | DRAFT |
| **Approval date** |  |
| **API consumers** | Mobile Channel, Internet Banking |
| **Business logic** | 1. ESB will receive the bill payment request from the channel 2. ESB will validate the request signature 3. ESB will query the biller to fetch the bill amount 4. ESB will debit the customer account in CBS and credit the biller collection account 5. ESB will notify the customer by SMS after successful payment |
| **Acceptance criteria** | 1. ESB should expose the bill payment API to the channels 2. ESB should verify the request signature before processing the payment 3. ESB should validate the request signature and reject invalid requests with 400 4. ESB should reverse the debit if the biller confirmation fails 5. ESB should send an SMS notification to the customer after payment 6. ESB should retry the biller confirmation 3 times with a 30 second interval |
| **Change log** |  |

## Non-functional Requirement

|  |  |
| --- | --- |
| **Anticipated Peak time TPS** |  |
| **Anticipated peak time duration** |  |
| **Any other (if required)** | Response time should be fast |

## Anticipated API resource

|  |  |
| --- | --- |
| **Protocol(RestFul,SOAP,File)** | HTTP RestFul |
| **Path parameters** | POST /bills/payments |
| **Request payload parameters****(Including Http status code)** | { "billerCode": "XXX", "billReference": "XXX", "accountNo": "XXX", "amount": "100", "currency": "TZS", "mobileNo": "XXX" } |
| **Response payload parameters****(Including Http status code)** | HTTP/1.1 202 Accepted { "paymentRef": "XXX", "status": "RECEIVED" } |
| **Error responses(Including Http status code, Payload)** | HTTP/1.1 400 Bad Request { "errorCode": "string", "errorDescription": "string" } |

## Backend details

### RestFull/Soap backend

|  |  |
| --- | --- |
| **Name of the backend** |  |
| **Protocol ** |  |
| **URL/IP** |  |
| **security details** |  |

## Mappings:

### Request Mapping:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **Source** | **Source field** | **Target** | **Target Field** | **Optional/Mandatory** | **Remarks** |
| API Request Body | billerCode | Biller | BillerId | Mandatory | |
| API Request Body | billReference | Biller | ControlNumber | Mandatory | |
| API Request Body | accountNo | CBS | DebitAccount | Mandatory | |
| API Request Body | amount | CBS | Amount | Mandatory | |
| API Request Body | currency | CBS |  | Mandatory | |
