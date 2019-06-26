# Index

## Data guiding principles

### Harm
+ Exposure
  + Will exposure of this data to other parties cause harm?
    + Implication: Decide whether the data should be used at all. Data must be protected at rest and in transit. Unauthorized users must be prevented from accessing the data. The data requires encryption at rest and in transit. Do not use data if using it creates a risk of harm.
+ Combination with other data
  + Will combining this data with other data in the project create information that if exposed would cause harm?
    + Implication: see exposure above.
  + Will combining this data with other data in future projects create information that if exposed would cause harm?
    + Implication: The data must be governed by a terms of use that prevents it from being used for other purposes. Do not allow it to be redistributed for other purposes.
+ Sensitivity
  + Does the data have personally identifiable information (PII)?
    +Implication: Data with PII is sensitive and must be protected. The use of personally identifiable data must respect the people it represents.

### Permission
+ Permission
  + Are we permitted to use the data?
    + Implication: Ensure there is permission to use the data. Document it. Understand the conditions of usage. Validate during the project that the project continues to respect the conditions of usage.

### Third party use
+ Third party use (examples: company, educational institution, government body)
  + Will a third party have access to the data as a result of a licence agreement?
    + Implication: Ensure the terms of usage would not lead to potential harm or non-permitted use. Avoid the use of cloud storage by companies such as Google, Microsoft, etc. in cases where data is sensitive. (The use of Google Colaboratory, Azure's API, Binder, GitHub should be considered vs a purely local environment.)
  + Will a third party have access to the data as a result of a geography?
    + Implication: Can the agency of a national government access the data because it transits or is stored in their country? Consider where data is stored and how is it transmitted as part of the project.
     
### Loss
+ Loss/Destruction
  + If the data was lost, destroyed or degraded would it result in a loss of an asset?
    + Implication: Ensure the data is backedup. Test that data can be restored to its original state. Ensure backups are protected.
  
     


