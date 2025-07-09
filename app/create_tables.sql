-- Create the Schedule table
CREATE TABLE dbo.Schedule (
    ScheduleID INT IDENTITY(1,1) PRIMARY KEY,
    [date] DATE NOT NULL,
    [time] TIME NOT NULL,
    position VARCHAR(100) NOT NULL,
    available BIT NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT GETDATE(),
    updated_at DATETIME DEFAULT GETDATE()
);

-- Insert some sample data
INSERT INTO dbo.Schedule ([date], [time], position, available)
VALUES 
    ('2024-03-20', '09:00', 'Python Dev', 0),
    ('2024-03-20', '10:00', 'Python Dev', 0),
    ('2024-03-20', '11:00', 'Python Dev', 0),
    ('2024-03-20', '14:00', 'Python Dev', 0),
    ('2024-03-20', '15:00', 'Python Dev', 0); 