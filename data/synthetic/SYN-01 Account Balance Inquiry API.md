# SYN-01 Account Balance Inquiry API

This template used for gathering use case details from the client.

## General Requirement

|  |  |
| --- | --- |
| **Use-case Name** | Partner Channel - Account Balance Inquiry API |
| **Use-case description** | This API allows the partner mobile channel to retrieve the available balance of a customer account from the core banking system |
| **Created by** | BA 2 |
| **Contact person** | BA 2 |
| **Approval Status** | BANK_X_APPROVED |
| **Approved By** | Product owner |
| **Approval date** |  |
| **API consumers** | Mobile Channel |
| **Business logic** | 1. ESB will receive the balance inquiry request from the mobile channel 2. ESB will validate the request headers and the JWT token 3. ESB will call the CBS account inquiry service to fetch the balance 4. ESB will transform the CBS response and return it to the channel 5. ESB will log the request and response in the audit database |
| **Acceptance criteria** | 1. ESB should expose a REST API for balance inquiry 2. ESB should validate the JWT token and return 401 if the token is invalid 3. ESB should verify the access token before calling the core banking system 4. ESB should respond quickly to the channel 5. ESB should retry the CBS call in case of timeout 6. Proper error handling |
| **Change log** | On Jan 10 2025 added accountType field |

## Non-functional Requirement

|  |  |
| --- | --- |
| **Anticipated Peak time TPS** | 50 |
| **Anticipated peak time duration** |  |
| **Any other (if required)** | The API should be highly available and user friendly |

## Anticipated API resource

|  |  |
| --- | --- |
| **Protocol(RestFul,SOAP,File)** | HTTP RestFul |
| **Header parameters** | Authorization: Bearer <token> Content-Type: application/json |
| **Query parameters** |  |
| **Path parameters** | GET /accounts/{accountNo}/balance |
| **Request payload parameters****(Including Http status code)** | { "accountNo": "XXX", "accountType": "SAVINGS", "channelRef": "XXX" } |
| **Response payload parameters****(Including Http status code)** | HTTP/1.1 200 OK { "accountNo": "XXX", "availableBalance": "1000.00", "currency": "TZS" } |
| **Error responses(Including Http status code, Payload)** |  |

## Backend details

### RestFull/Soap backend

|  |  |
| --- | --- |
| **Name of the backend** | CBS Account Inquiry Service |
| **Protocol ** | SOAP |
| **URL/IP** | https://cbs.example.internal/account |
| **security details** |  |
| **Sample Request ** |  |
| **Success Response** |  |
| **Validations** |  |
| **Additional Comments** |  |

## DBMS System

|  |  |
| --- | --- |
| **Name of the DB** |  |
| **Type of the DB ** | SQL/NoSQL |
| **Vendor** | oracle,Mysql |
| **URL/s** |  |
| **Security details** |  |

## Mappings:

### Request Mapping:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **Source** | **Source field** | **Target** | **Target Field** | **Optional/Mandatory** | **Remarks** |
| API Request Path | accountNo | CBS | AccountNumber | Mandatory | |
| API Request Body | channelRef | CBS | ReferenceId | | Channel reference |

### Response Mapping

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **Source** | **Source field** | **Target** | **Target Field** | **Optional/Mandatory** | **Remarks** |
| CBS | AvailableBalance | API Response Body | availableBalance | Mandatory | |
| CBS | Currency | API Response Body | currency | Mandatory | |
